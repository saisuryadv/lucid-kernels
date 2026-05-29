"""Correctness tests for LUCID forward substitution kernels.

Tests dV and dK against float64 autograd reference.
Usage: python test_correctness.py
"""
import torch
import sys

from lucid_kernels.cute import _forward_sub_fwd, _forward_sub_bwd


def build_L(K_bshd, block_size):
    """Build full N×N lower triangular matrix L from block keys."""
    B, S, H, D = K_bshd.shape
    T = S // block_size
    BS = block_size
    N = S
    L_full = torch.zeros(B, H, N, N, device=K_bshd.device, dtype=K_bshd.dtype)
    for b in range(B):
        for h in range(H):
            K_flat = K_bshd[b, :, h, :]
            gram = K_flat @ K_flat.T
            E = torch.exp(gram)
            for i in range(T):
                for j in range(T):
                    block = E[i*BS:(i+1)*BS, j*BS:(j+1)*BS]
                    if j < i:
                        L_full[b, h, i*BS:(i+1)*BS, j*BS:(j+1)*BS] = block
                    elif j == i:
                        L_full[b, h, i*BS:(i+1)*BS, j*BS:(j+1)*BS] = torch.tril(block)
    return L_full


def forward_sub_autograd(K_bshd, V_bshd, block_size):
    """Forward substitution using full matrix solve (autograd-compatible)."""
    B, S, H, D = K_bshd.shape
    L = build_L(K_bshd, block_size)
    V_bhsd = V_bshd.permute(0, 2, 1, 3)
    Vp_bhsd = torch.linalg.solve_triangular(L, V_bhsd, upper=False)
    return Vp_bhsd.permute(0, 2, 1, 3)


def test_combined_kernel():
    """Test combined dV+dK kernel against float64 autograd reference."""
    device = "cuda"
    print("=== LUCID Kernel Correctness Tests ===\n")

    configs = [
        (1, 1, 256, 128, 128),   # T=2
        (1, 1, 384, 128, 128),   # T=3
        (1, 1, 512, 128, 128),   # T=4
        (1, 1, 1024, 128, 128),  # T=8
    ]

    all_pass = True
    for batch, heads, seq_len, head_dim, block_size in configs:
        T = seq_len // block_size
        label = f"B={batch}, H={heads}, S={seq_len}, D={head_dim}, T={T}"

        # Float64 autograd reference
        K_raw = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=torch.float64) * 0.1
        V_raw = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=torch.float64)
        K_raw.requires_grad_(True)
        V_raw.requires_grad_(True)
        Vp = forward_sub_autograd(K_raw, V_raw, block_size)
        dVp = torch.randn_like(Vp)
        loss = (Vp * dVp).sum()
        loss.backward()
        dK_auto = K_raw.grad.clone()
        dV_auto = V_raw.grad.clone()

        # bf16 kernel
        K_bf16 = K_raw.detach().bfloat16()
        V_bf16 = V_raw.detach().bfloat16()
        dVp_bf16 = dVp.bfloat16()
        Vp_kern = _forward_sub_fwd(K_bf16, V_bf16, block_size=block_size, use_diag_solve=True)
        dV_kern, dK_kern = _forward_sub_bwd(K_bf16, Vp_kern, dVp_bf16,
                                             block_size=block_size, use_diag_solve=True,
                                             use_combined_kernel=True)
        torch.cuda.synchronize()

        dV_err = (dV_kern.float() - dV_auto.float()).norm() / dV_auto.float().norm()
        dK_err = (dK_kern.float() - dK_auto.float()).norm() / dK_auto.float().norm()
        dv_ok = dV_err < 0.05
        dk_ok = dK_err < 0.05
        status = "PASS" if (dv_ok and dk_ok) else "FAIL"
        if not (dv_ok and dk_ok):
            all_pass = False

        print(f"  {label}: dV={dV_err:.4e} dK={dK_err:.4e}  {status}")

    print(f"\n{'All tests passed!' if all_pass else 'SOME TESTS FAILED!'}")
    return all_pass


if __name__ == "__main__":
    ok = test_combined_kernel()
    sys.exit(0 if ok else 1)
