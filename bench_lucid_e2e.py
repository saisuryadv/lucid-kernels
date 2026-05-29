"""End-to-end LUCID attention layer benchmark: throughput and peak memory.

Metrics:
  - Inference throughput (forward-only, tokens/sec)
  - Training throughput (forward+backward, tokens/sec)
  - Peak GPU memory (MB)

Sweeps sequence lengths 512 to 128K across multiple head configs.

Usage:
    python bench_lucid_e2e.py
"""

import sys
import types
import torch
import time
import gc

# Mock C extension modules
for mod_name in ('flash_attn_2_cuda', 'flash_attn_3_cuda'):
    sys.modules[mod_name] = types.ModuleType(mod_name)

from lucid_kernels.cute.forward_sub_interface import (
    _forward_sub_fwd, _forward_sub_bwd,
)
try:
    from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False


class LUCIDForwardSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, block_size=128):
        v_prime = _forward_sub_fwd(k, v, use_diag_solve=True)
        ctx.save_for_backward(k, v_prime)
        ctx.block_size = block_size
        return v_prime

    @staticmethod
    def backward(ctx, dv_prime):
        k, v_prime = ctx.saved_tensors
        dv, dk = _forward_sub_bwd(k, v_prime, dv_prime,
                                   block_size=ctx.block_size,
                                   use_diag_solve=True,
                                   use_combined_kernel=True)
        return dk, dv, None


class FA3Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out, lse = _flash_attn_fwd(q, k, v, causal=True, return_lse=True)
        ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        _flash_attn_bwd(q, k, v, out, dout, lse, causal=True)
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)


def bench_time(fn, warmup=3, iters=10):
    """Return median time in seconds."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def run_lucid(batch, seqlen, num_heads, head_dim, device, dtype):
    """Run LUCID fwd, fwd+bwd, measure throughput + memory."""
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()

    def make_kv():
        k = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype) * 0.1
        v = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        return k, v

    # Inference (forward-only)
    k, v = make_kv()
    fwd_sec = bench_time(lambda: _forward_sub_fwd(k, v, use_diag_solve=True))
    del k, v

    # Training (forward + backward via autograd)
    torch.cuda.reset_peak_memory_stats()
    def train_step():
        k = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True) * 0.1
        v = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        out = LUCIDForwardSub.apply(k, v, 128)
        out.sum().backward()

    train_sec = bench_time(train_step)
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    tokens = batch * seqlen * num_heads
    infer_tps = tokens / fwd_sec
    train_tps = tokens / train_sec

    return fwd_sec * 1000, train_sec * 1000, infer_tps, train_tps, peak_mem_mb


def run_fa3(batch, seqlen, num_heads, head_dim, device, dtype):
    """Run FA3 fwd, fwd+bwd, measure throughput + memory."""
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()

    q = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype)

    # Inference (forward-only)
    fwd_sec = bench_time(lambda: _flash_attn_fwd(q, k, v, causal=True))
    del q, k, v

    # Training (forward + backward via autograd)
    torch.cuda.reset_peak_memory_stats()
    def train_step():
        q = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        out = FA3Attention.apply(q, k, v)
        out.sum().backward()

    train_sec = bench_time(train_step)
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    tokens = batch * seqlen * num_heads
    infer_tps = tokens / fwd_sec
    train_tps = tokens / train_sec

    return fwd_sec * 1000, train_sec * 1000, infer_tps, train_tps, peak_mem_mb


def fmt_tps(tps):
    """Format tokens/sec as human-readable."""
    if tps >= 1e9:
        return f"{tps/1e9:.1f}B"
    elif tps >= 1e6:
        return f"{tps/1e6:.1f}M"
    elif tps >= 1e3:
        return f"{tps/1e3:.0f}K"
    return f"{tps:.0f}"


def main():
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1

    gpu_name = torch.cuda.get_device_name()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"{'='*100}")
    print(f"LUCID End-to-End Attention Layer Benchmark")
    print(f"{'='*100}")
    print(f"Hardware: {gpu_name}, {gpu_mem_gb:.0f} GB HBM")
    print(f"Software: PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Config:   batch={batch}, dtype=bf16, use_diag_solve=True")
    print(f"Metrics:  Inference throughput (fwd-only), Training throughput (fwd+bwd)")
    print(f"          Peak GPU memory during training step")
    print()

    seqlens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    configs = [
        ("1h-D128", 1, 128),
        ("4h-D128", 4, 128),
        ("8h-D128", 8, 128),
    ]

    for config_name, num_heads, head_dim in configs:
        print(f"{'='*100}")
        print(f"Config: {config_name} (heads={num_heads}, D={head_dim})")
        print(f"{'='*100}")
        hdr = (f"{'Seqlen':>8} | {'LUCID Infer':>12} {'LUCID Train':>12} {'LUCID Mem':>10} |"
               f" {'FA3 Infer':>12} {'FA3 Train':>12} {'FA3 Mem':>10} |"
               f" {'Fwd ratio':>10} {'Train ratio':>12}")
        sub = (f"{'':>8} | {'(tok/s)':>12} {'(tok/s)':>12} {'(MB)':>10} |"
               f" {'(tok/s)':>12} {'(tok/s)':>12} {'(MB)':>10} |"
               f" {'LUCID/FA3':>10} {'LUCID/FA3':>12}")
        print(hdr)
        print(sub)
        print("-" * len(hdr))

        for seqlen in seqlens:
            try:
                luc_fwd_ms, luc_train_ms, luc_infer_tps, luc_train_tps, luc_mem = \
                    run_lucid(batch, seqlen, num_heads, head_dim, device, dtype)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    luc_fwd_ms = luc_train_ms = luc_infer_tps = luc_train_tps = luc_mem = None
                else:
                    raise

            try:
                fa3_fwd_ms, fa3_train_ms, fa3_infer_tps, fa3_train_tps, fa3_mem = \
                    run_fa3(batch, seqlen, num_heads, head_dim, device, dtype)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or "OOM" in str(e):
                    fa3_fwd_ms = fa3_train_ms = fa3_infer_tps = fa3_train_tps = fa3_mem = None
                else:
                    raise

            if luc_infer_tps and fa3_infer_tps:
                fwd_ratio = luc_fwd_ms / fa3_fwd_ms
                train_ratio = luc_train_ms / fa3_train_ms
                print(f"{seqlen:>8} | {fmt_tps(luc_infer_tps):>12} {fmt_tps(luc_train_tps):>12} {luc_mem:>10.0f} |"
                      f" {fmt_tps(fa3_infer_tps):>12} {fmt_tps(fa3_train_tps):>12} {fa3_mem:>10.0f} |"
                      f" {fwd_ratio:>9.1f}x {train_ratio:>11.1f}x")
            elif luc_infer_tps:
                print(f"{seqlen:>8} | {fmt_tps(luc_infer_tps):>12} {fmt_tps(luc_train_tps):>12} {luc_mem:>10.0f} |"
                      f" {'OOM':>12} {'OOM':>12} {'OOM':>10} |"
                      f" {'---':>10} {'---':>12}")
            elif fa3_infer_tps:
                print(f"{seqlen:>8} | {'OOM':>12} {'OOM':>12} {'OOM':>10} |"
                      f" {fmt_tps(fa3_infer_tps):>12} {fmt_tps(fa3_train_tps):>12} {fa3_mem:>10.0f} |"
                      f" {'---':>10} {'---':>12}")
            else:
                print(f"{seqlen:>8} | {'OOM':>12} {'OOM':>12} {'OOM':>10} |"
                      f" {'OOM':>12} {'OOM':>12} {'OOM':>10} |"
                      f" {'---':>10} {'---':>12}")

            sys.stdout.flush()

        print()


if __name__ == "__main__":
    main()
