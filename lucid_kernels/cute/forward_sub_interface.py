# Forward Substitution interface — upstream FA3 patterns.
#
# Provides _forward_sub_fwd(k, v, block_size) which computes tiled
# forward substitution V' using the modified FA3 SM90 kernel.
#
# Key differences from LUCID interface:
#   - Uses to_cute_tensor() instead of manual from_dlpack
#   - Uses cute.compile(..., options="--enable-tvm-ffi")
#   - Uses get_jit_cache("forward_sub") for persistent disk cache
#   - Invokes compiled function with raw torch tensors (TVM-FFI handles conversion)

import torch
from typing import Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from lucid_kernels.cute.cute_dsl_utils import to_cute_tensor
from lucid_kernels.cute.cache_utils import get_jit_cache
from lucid_kernels.cute.testing import is_fake_mode
from lucid_kernels.cute.forward_sub_fwd import ForwardSubSm90


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _compute_diag_inv(k, block_size):
    """Pre-compute inv(tril(exp(K_i @ K_i^T))) for each block i.

    Args:
        k: Key tensor (batch, seqlen, heads_kv, head_dim) in BSHD layout.
        block_size: Tile size (must equal head_dim for WGMMA compatibility).

    Returns:
        diag_inv: (batch, seqlen, heads_kv, block_size) tensor in bf16/fp16,
                  where each block i's BS×BS inverse is stored in the
                  seqlen dimension at positions [i*BS:(i+1)*BS].
    """
    B, S, H, D = k.shape
    T = S // block_size
    BS = block_size
    # Reshape to (B*H*T, BS, D)
    k_blocks = k.reshape(B, T, BS, H, D).permute(0, 3, 1, 2, 4).reshape(-1, BS, D).float()
    KKT = k_blocks @ k_blocks.transpose(-1, -2)  # (B*H*T, BS, BS)
    M = torch.tril(torch.exp(KKT))
    I = torch.eye(BS, device=k.device, dtype=torch.float32).expand_as(M)
    diag_inv = torch.linalg.solve_triangular(M, I, upper=False)
    # Reshape to (B, H, T, BS, BS) then to (B, T*BS, H, BS) = BSHD with D=BS
    diag_inv = diag_inv.reshape(B, H, T, BS, BS).permute(0, 2, 3, 1, 4)
    return diag_inv.reshape(B, T * BS, H, BS).to(k.dtype).contiguous()


def _forward_sub_fwd(
    k: torch.Tensor,           # (batch, seqlen, heads_kv, head_dim)
    v: torch.Tensor,           # (batch, seqlen, heads_kv, head_dim)
    block_size: int = 128,
    window_size: int = None,   # None = full causality, int = last W blocks
    no_sync: bool = False,     # Skip flag synchronization (for profiling)
    instrument: bool = False,  # Record per-CTA timestamps
    use_diag_solve: bool = True,  # Apply diagonal block solve
) -> torch.Tensor:
    """Compute tiled forward substitution using modified FA3 SM90 kernel.

    V'_i = V_i - sum_{j<i} exp(K_i @ K_j^T) @ V'_j

    With window_size=W, truncates to last W blocks:
    V'_i = V_i - sum_{j=max(0,i-W)}^{i-1} exp(K_i @ K_j^T) @ V'_j

    Args:
        k: Key tensor in BSHD layout (batch, seqlen, heads_kv, head_dim)
        v: Value tensor in BSHD layout (batch, seqlen, heads_kv, head_dim)
        block_size: Tile size (must divide seqlen)
        window_size: Number of previous blocks to correct against (None = all)

    Returns:
        v_prime: Output tensor (same shape as v)
    """
    k, v = [maybe_contiguous(t) for t in (k, v)]

    batch_size, seqlen, num_head_kv, head_dim = k.shape
    head_dim_v = v.shape[-1]
    assert v.shape == (batch_size, seqlen, num_head_kv, head_dim_v)
    assert seqlen % block_size == 0, f"seqlen ({seqlen}) must be divisible by block_size ({block_size})"
    assert k.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert k.dtype == v.dtype, "k and v must have the same dtype"
    assert all(t.is_cuda for t in (k, v)), "inputs must be on CUDA device"
    alignment = 16 // k.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"

    T = seqlen // block_size
    device = k.device

    # Allocate output V' buffer (also read by kernel as streaming V'_j)
    v_prime = torch.empty_like(v)

    # Allocate synchronization flags: one per (head*batch, block)
    flags = torch.zeros(batch_size * num_head_kv, T, dtype=torch.int32, device=device)

    # Allocate debug timestamps: (T, 8) int64
    timestamps = torch.zeros(T, 8, dtype=torch.int64, device=device)

    m_block_size = block_size
    n_block_size = block_size
    num_threads = 384

    if use_diag_solve:
        assert block_size == head_dim, (
            f"diag_solve requires block_size == head_dim, got {block_size} != {head_dim}"
        )
        diag_inv = _compute_diag_inv(k, block_size)
    else:
        # Dummy tensor (never accessed by kernel, but needed for TMA descriptor creation)
        diag_inv = torch.zeros(batch_size, seqlen, num_head_kv, block_size,
                               device=device, dtype=k.dtype)

    compile_key = (
        k.dtype, head_dim, head_dim_v, m_block_size, n_block_size,
        num_threads, window_size, no_sync, instrument, use_diag_solve,
    )
    if compile_key not in _forward_sub_fwd.compile_cache:
        # Convert to CuTe tensors for compilation
        k_tensor, v_tensor, vprime_tensor, vorig_tensor = [
            to_cute_tensor(t) for t in (k, v, v_prime, v)
        ]
        flags_tensor = to_cute_tensor(flags, assumed_align=4)
        timestamps_tensor = to_cute_tensor(timestamps, assumed_align=8)
        diag_inv_tensor = to_cute_tensor(diag_inv)

        fwd_sub = ForwardSubSm90(
            cutlass.Float16 if k.dtype == torch.float16 else cutlass.BFloat16,
            head_dim,
            head_dim_v,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_stages=2,
            num_threads=num_threads,
            window_size=window_size,
            no_sync=no_sync,
            instrument=instrument,
            use_diag_solve=use_diag_solve,
        )
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _forward_sub_fwd.compile_cache[compile_key] = cute.compile(
            fwd_sub,
            k_tensor, v_tensor, vprime_tensor, flags_tensor, vorig_tensor,
            timestamps_tensor, diag_inv_tensor, current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _forward_sub_fwd.compile_cache[compile_key](
            k.detach(), v.detach(), v_prime, flags, v.detach(),
            timestamps, diag_inv, current_stream,
        )

    if instrument:
        return v_prime, timestamps
    return v_prime


_forward_sub_fwd.compile_cache = get_jit_cache("forward_sub")


# =========================================================================
# Backward pass
# =========================================================================

def _get_backward_cls():
    from lucid_kernels.cute.forward_sub_bwd import BackwardSubSm90
    return BackwardSubSm90


def _compute_diag_inv_transpose(k, block_size):
    """Pre-compute inv(triu(exp(K_i @ K_i^T)))^T for each block i (backward diagonal solve).

    The backward solve needs inv(L_diag^T) where L_diag = tril(exp(K_i@K_i^T)).
    inv(L_diag^T) = inv(L_diag)^T, so we compute inv(tril(exp(K_i@K_i^T))) and transpose.
    """
    B, S, H, D = k.shape
    T = S // block_size
    BS = block_size
    k_blocks = k.reshape(B, T, BS, H, D).permute(0, 3, 1, 2, 4).reshape(-1, BS, D).float()
    KKT = k_blocks @ k_blocks.transpose(-1, -2)
    M = torch.tril(torch.exp(KKT))
    I = torch.eye(BS, device=k.device, dtype=torch.float32).expand_as(M)
    diag_inv = torch.linalg.solve_triangular(M, I, upper=False)
    diag_inv_t = diag_inv.transpose(-1, -2).contiguous()
    diag_inv_t = diag_inv_t.reshape(B, H, T, BS, BS).permute(0, 2, 3, 1, 4)
    return diag_inv_t.reshape(B, T * BS, H, BS).to(k.dtype).contiguous()


def _forward_sub_bwd_dv(
    k: torch.Tensor,
    dv_prime: torch.Tensor,
    block_size: int = 128,
    use_diag_solve: bool = True,
    instrument: bool = False,
) -> torch.Tensor:
    """Step 1 of backward: solve for dV via upper triangular solve in reverse order."""
    k, dv_prime = [maybe_contiguous(t) for t in (k, dv_prime)]

    batch_size, seqlen, num_head_kv, head_dim = k.shape
    head_dim_v = dv_prime.shape[-1]
    assert dv_prime.shape == (batch_size, seqlen, num_head_kv, head_dim_v)
    assert seqlen % block_size == 0
    assert k.dtype in [torch.float16, torch.bfloat16]
    assert k.dtype == dv_prime.dtype
    assert all(t.is_cuda for t in (k, dv_prime))

    T = seqlen // block_size
    device = k.device

    dv = torch.empty_like(dv_prime)
    flags = torch.zeros(batch_size * num_head_kv, T, dtype=torch.int32, device=device)
    timestamps = torch.zeros(T, 8, dtype=torch.int64, device=device)

    m_block_size = block_size
    n_block_size = block_size
    num_threads = 384
    dtype = cutlass.Float16 if k.dtype == torch.float16 else cutlass.BFloat16

    if use_diag_solve:
        assert block_size == head_dim
        diag_inv_t = _compute_diag_inv_transpose(k, block_size)
    else:
        diag_inv_t = torch.zeros(batch_size, seqlen, num_head_kv, block_size,
                                 device=device, dtype=k.dtype)

    compile_key = ("bwd_dv", k.dtype, head_dim, head_dim_v, m_block_size, n_block_size,
                   num_threads, instrument, use_diag_solve)
    if compile_key not in _forward_sub_bwd_dv.compile_cache:
        k_tensor, dvprime_tensor, dv_tensor, dvprime_orig_tensor = [
            to_cute_tensor(t) for t in (k, dv_prime, dv, dv_prime)
        ]
        flags_tensor = to_cute_tensor(flags, assumed_align=4)
        timestamps_tensor = to_cute_tensor(timestamps, assumed_align=8)
        diag_inv_t_tensor = to_cute_tensor(diag_inv_t)

        bwd_sub = _get_backward_cls()(
            dtype, head_dim, head_dim_v,
            m_block_size=m_block_size, n_block_size=n_block_size,
            num_stages=2, num_threads=num_threads,
            instrument=instrument, use_diag_solve=use_diag_solve,
        )
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _forward_sub_bwd_dv.compile_cache[compile_key] = cute.compile(
            bwd_sub, k_tensor, dvprime_tensor, dv_tensor, flags_tensor, dvprime_orig_tensor,
            timestamps_tensor, diag_inv_t_tensor, current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _forward_sub_bwd_dv.compile_cache[compile_key](
            k.detach(), dv_prime.detach(), dv, flags, dv_prime.detach(),
            timestamps, diag_inv_t, current_stream,
        )

    if instrument:
        return dv, timestamps
    return dv


_forward_sub_bwd_dv.compile_cache = get_jit_cache("forward_sub_bwd_dv")


def _forward_sub_bwd_combined(
    k: torch.Tensor,
    v_prime: torch.Tensor,
    dv_prime: torch.Tensor,
    block_size: int = 128,
    use_diag_solve: bool = True,
    instrument: bool = False,
) -> tuple:
    """Combined dV + dK kernel (BackwardSubSm90 with compute_dk=True).

    Returns (dv, dk) where dk contains off-diagonal contributions only.
    """
    k, v_prime, dv_prime = [maybe_contiguous(t) for t in (k, v_prime, dv_prime)]

    batch_size, seqlen, num_head_kv, head_dim = k.shape
    head_dim_v = dv_prime.shape[-1]
    assert dv_prime.shape == (batch_size, seqlen, num_head_kv, head_dim_v)
    assert v_prime.shape == (batch_size, seqlen, num_head_kv, head_dim_v)
    assert seqlen % block_size == 0
    assert k.dtype in [torch.float16, torch.bfloat16]
    assert k.dtype == dv_prime.dtype == v_prime.dtype
    assert all(t.is_cuda for t in (k, dv_prime, v_prime))

    T = seqlen // block_size
    device = k.device

    dv = torch.empty_like(dv_prime)
    dk = torch.zeros(batch_size, seqlen, num_head_kv, head_dim, device=device, dtype=k.dtype)
    flags = torch.zeros(batch_size * num_head_kv, T, dtype=torch.int32, device=device)
    timestamps = torch.zeros(T, 8, dtype=torch.int64, device=device)

    m_block_size = block_size
    n_block_size = block_size
    num_threads = 384
    dtype = cutlass.Float16 if k.dtype == torch.float16 else cutlass.BFloat16

    if use_diag_solve:
        assert block_size == head_dim
        diag_inv_t = _compute_diag_inv_transpose(k, block_size)
    else:
        diag_inv_t = torch.zeros(batch_size, seqlen, num_head_kv, block_size,
                                 device=device, dtype=k.dtype)

    compile_key = ("bwd_combined", k.dtype, head_dim, head_dim_v, m_block_size, n_block_size,
                   num_threads, instrument, use_diag_solve)
    if compile_key not in _forward_sub_bwd_combined.compile_cache:
        k_tensor, dvprime_tensor, dv_tensor, dvprime_orig_tensor = [
            to_cute_tensor(t) for t in (k, dv_prime, dv, dv_prime)
        ]
        flags_tensor = to_cute_tensor(flags, assumed_align=4)
        timestamps_tensor = to_cute_tensor(timestamps, assumed_align=8)
        diag_inv_t_tensor = to_cute_tensor(diag_inv_t)
        vprime_tensor = to_cute_tensor(v_prime)
        dk_tensor = to_cute_tensor(dk)

        bwd_sub = _get_backward_cls()(
            dtype, head_dim, head_dim_v,
            m_block_size=m_block_size, n_block_size=n_block_size,
            num_stages=2, num_threads=num_threads,
            instrument=instrument, use_diag_solve=use_diag_solve,
            compute_dk=True,
        )
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _forward_sub_bwd_combined.compile_cache[compile_key] = cute.compile(
            bwd_sub, k_tensor, dvprime_tensor, dv_tensor, flags_tensor, dvprime_orig_tensor,
            timestamps_tensor, diag_inv_t_tensor, current_stream,
            vprime_tensor, dk_tensor,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _forward_sub_bwd_combined.compile_cache[compile_key](
            k.detach(), dv_prime.detach(), dv, flags, dv_prime.detach(),
            timestamps, diag_inv_t, current_stream,
            v_prime.detach(), dk,
        )

    if instrument:
        return dv, dk, timestamps
    return dv, dk


_forward_sub_bwd_combined.compile_cache = get_jit_cache("forward_sub_bwd_combined")


def _forward_sub_bwd(
    k: torch.Tensor,
    v_prime: torch.Tensor,
    dv_prime: torch.Tensor,
    block_size: int = 128,
    use_diag_solve: bool = True,
    use_combined_kernel: bool = False,
) -> tuple:
    """Full backward pass for forward substitution.

    Returns (dv, dk): gradients w.r.t. V and K.
    """
    if not use_combined_kernel:
        dv = _forward_sub_bwd_dv(k, dv_prime, block_size=block_size, use_diag_solve=use_diag_solve)
        dK = _backward_dk_reference(k, v_prime, dv, block_size)
        return dv, dK

    dv, dK = _forward_sub_bwd_combined(k, v_prime, dv_prime,
                                        block_size=block_size,
                                        use_diag_solve=use_diag_solve)
    dK = _add_diagonal_dk_correction(k, v_prime, dv, dK, block_size)
    return dv, dK


def _add_diagonal_dk_correction(k, v_prime, dv, dk, block_size):
    """Add diagonal (j=i) dK contributions computed on host.

    The kernel computes off-diagonal dK. The diagonal blocks need:
      dS_diag[i] = tril(-dV_i @ V'_i^T * exp(K_i @ K_i^T))
      dS_sym_diag = dS_diag + dS_diag^T
      dK_diag_correction[i] = dS_sym_diag @ K_i
    """
    B, S, H, D = k.shape
    T = S // block_size
    BS = block_size

    K = k.permute(0, 2, 1, 3).reshape(B, H, T, BS, D).float()
    Vp = v_prime.permute(0, 2, 1, 3).reshape(B, H, T, BS, D).float()
    dV = dv.permute(0, 2, 1, 3).reshape(B, H, T, BS, D).float()

    KKT_diag = K @ K.transpose(-1, -2)
    E_diag = torch.exp(KKT_diag)
    G_diag = -(dV @ Vp.transpose(-1, -2)) * E_diag
    G_diag = torch.tril(G_diag)
    dS_sym_diag = G_diag + G_diag.transpose(-1, -2)
    dK_corr = dS_sym_diag @ K

    dK_corr = dK_corr.reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous().to(k.dtype)
    return dk + dK_corr


def _backward_dk_reference(k, v_prime, dv, block_size):
    """Batched blocked dK computation (host reference).

    dK = (dS + dS^T) @ K  where dS = block_tril(-dV @ V'^T * exp(K @ K^T))
    """
    B, S, H, D = k.shape
    T = S // block_size
    BS = block_size

    K = k.permute(0, 2, 1, 3).reshape(B, H, T, BS, D).float()
    Vp = v_prime.permute(0, 2, 1, 3).reshape(B, H, T, BS, D).float()
    dV = dv.permute(0, 2, 1, 3).reshape(B, H, T, BS, D).float()

    E = torch.exp(K.unsqueeze(3) @ K.unsqueeze(2).transpose(-1, -2))
    M = -(dV.unsqueeze(3) @ Vp.unsqueeze(2).transpose(-1, -2)) * E

    block_mask_upper = torch.triu(torch.ones(T, T, device=k.device, dtype=torch.bool), diagonal=1)
    M[:, :, block_mask_upper] = 0
    for t in range(T):
        M[:, :, t, t] = torch.tril(M[:, :, t, t])

    dS_sym = M + M.permute(0, 1, 3, 2, 5, 4)
    dK = torch.einsum('bhijsr,bhjrd->bhisd', dS_sym, K)
    dK = dK.reshape(B, H, S, D).permute(0, 2, 1, 3).contiguous()
    return dK.to(k.dtype)
