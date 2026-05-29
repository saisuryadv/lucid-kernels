# LUCID Kernels

Custom SM90 (Hopper) CuTe DSL kernels for LUCID's block-triangular forward substitution:

```
V'_i = inv(tril(exp(K_i @ K_i^T))) @ (V_i - Σ_{j<i} exp(K_i @ K_j^T) @ V'_j)
```

Forward and backward (dV + dK) kernels with inter-CTA flag synchronization, TMA async pipelines, and WGMMA.

## Installation

```bash
pip install -e .
```

**Requirements:** NVIDIA Hopper GPU (SM90), CUDA 12.4+, PyTorch 2.4+, `nvidia-cutlass-dsl>=4.4`, `quack-kernels>=0.2`

## Usage

```python
from lucid_kernels.cute import _forward_sub_fwd, _forward_sub_bwd

# Forward: V' = L^{-1} V
v_prime = _forward_sub_fwd(k, v, use_diag_solve=True)

# Backward: dV, dK
dv, dk = _forward_sub_bwd(k, v_prime, dv_prime,
                           use_diag_solve=True,
                           use_combined_kernel=True)
```

Tensors must be bf16/fp16, BSHD layout `(batch, seqlen, heads, head_dim)`, with `head_dim=128` and `seqlen` divisible by 128.

## Correctness

```bash
python test_correctness.py
```

Verified against float64 autograd reference: dV <1% error, dK <2% error (T=2..8).

## Benchmarks

### Kernel Benchmark (raw kernel timing)

```bash
python bench_lucid.py  # requires flash-attention upstream for FA3 baseline
```

### Full Model Benchmark (Gemma 3-4B-style)

```bash
python bench_lucid_model.py
```

Gemma 3-4B-style architecture: 34 layers (5 global with LUCID + 29 sliding window), GQA 2:1, head_dim=128.

**4 attention backends:** SDPA (FA2), FA3, LUCID (CuTe kernel), Naive (torch.linalg.solve_triangular)

#### Results (GH200 120GB, bf16, batch=1)

| Seqlen | SDPA Infer | FA3 Infer | LUCID Infer | Naive Infer | LUCID overhead | LUCID/Naive speedup |
|--------|-----------|-----------|-------------|-------------|---------------|-------------------|
| 512 | 20K tok/s | 19K tok/s | 18K tok/s | 15K tok/s | +10% | 1.2x |
| 2K | 81K | 80K | 70K | 19K | +10% | 3.7x |
| 8K | 124K | 127K | 118K | 7K | +5% | 17x |
| 32K | 82K | 85K | 78K | 2K | +5% | 39x |
| 64K | 55K | 57K | 53K | 865 | +4% | 61x |
| 128K | 32K | 34K | 31K | — | +4% | — |

**LUCID adds 4-10% overhead** vs softmax attention when applied to global layers only (5/34 layers).
**LUCID CuTe kernel is 17-61x faster** than naive `torch.linalg.solve_triangular`.

## Architecture

### Forward Kernel (`forward_sub_fwd.py`)
- Producer/consumer warp specialization (32 + 256 threads)
- 2-stage TMA async pipeline for K_j and V'_j streaming
- Flag synchronization for inter-CTA data dependencies
- Optional diagonal block solve via WGMMA

### Backward Kernel (`forward_sub_bwd.py`)
- **Phase 1** (upper triangle, sequential): dV via reverse triangular solve + upper-triangle dK
- **Phase 2** (lower triangle, parallel): lower-triangle dK accumulation
- K_i TMA reload between phases (sO/sQ alias fix)
- sVfixed R2S reload (solved dV_i for Phase 2 GEMM)
- Diagonal dK correction on host

## Files

| File | Description |
|------|-------------|
| `lucid_kernels/cute/forward_sub_fwd.py` | Forward kernel (ForwardSubSm90) |
| `lucid_kernels/cute/forward_sub_bwd.py` | Backward kernel (BackwardSubSm90) |
| `lucid_kernels/cute/forward_sub_interface.py` | Python interface |
| `test_correctness.py` | Correctness tests vs float64 reference |
| `bench_lucid.py` | Raw kernel benchmark |
| `bench_lucid_model.py` | Full model benchmark (Gemma 3-style) |
| `bench_lucid_e2e.py` | Attention layer benchmark |
