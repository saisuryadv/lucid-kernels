"""Benchmark LUCID forward+backward kernels vs upstream FA3.

All kernels from the upstream flash-attention repo.
Usage:
    python bench_lucid.py
"""

import torch
import time
import sys
import types

# Mock C extension modules (not needed for CuTe DSL kernels)
for mod_name in ('flash_attn_2_cuda', 'flash_attn_3_cuda'):
    sys.modules[mod_name] = types.ModuleType(mod_name)

from lucid_kernels.cute.forward_sub_interface import _forward_sub_fwd, _forward_sub_bwd_combined
# FA3 requires flash-attention-upstream repo
try:
    import sys, types
    for m in ("flash_attn_2_cuda", "flash_attn_3_cuda"): sys.modules[m] = types.ModuleType(m)
    from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

torch.manual_seed(42)
device = "cuda"
B, H, D, BS = 1, 1, 128, 128


def bench(fn, warmup=10, iters=50):
    """Return median time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    print(f"=== LUCID vs Upstream FA3 Kernel Benchmark ===")
    print(f"B={B}, H={H}, D={D}, BS={BS}, dtype=bf16")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"All times in microseconds.\n")

    hdr = (f"{'T':>4} {'S':>6} | {'FA3fwd':>8} {'FA3bwd':>8} |"
           f" {'LUCfwd':>8} {'LUCdVdK':>8} |"
           f" {'fwd/FA3':>8} {'bwd/FA3':>9}")
    print(hdr)
    print("-" * len(hdr))

    for T in [2, 4, 8, 16, 32, 64]:
        S = T * BS
        q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) * 0.1
        v = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16) * 0.1
        dVp = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)

        # FA3 upstream
        out, lse = _flash_attn_fwd(q, k, v, causal=True, return_lse=True)
        dout = torch.randn_like(out)
        torch.cuda.synchronize()
        _flash_attn_bwd(q, k, v, out, dout, lse, causal=True)
        torch.cuda.synchronize()
        fa3f = bench(lambda: _flash_attn_fwd(q, k, v, causal=True))
        fa3b = bench(lambda: _flash_attn_bwd(q, k, v, out, dout, lse, causal=True))

        # LUCID forward + backward (raw kernels, no diag_solve)
        Vp = _forward_sub_fwd(k, v, use_diag_solve=False)
        torch.cuda.synchronize()
        lucf = bench(lambda: _forward_sub_fwd(k, v, use_diag_solve=False))
        _forward_sub_bwd_combined(k, Vp, dVp, use_diag_solve=False)
        torch.cuda.synchronize()
        lucb = bench(lambda: _forward_sub_bwd_combined(k, Vp, dVp, use_diag_solve=False))

        print(f"{T:>4} {S:>6} | {fa3f:>7.0f}us {fa3b:>7.0f}us |"
              f" {lucf:>7.0f}us {lucb:>7.0f}us |"
              f" {lucf/fa3f:>7.1f}x {lucb/fa3b:>8.1f}x")


if __name__ == "__main__":
    main()
