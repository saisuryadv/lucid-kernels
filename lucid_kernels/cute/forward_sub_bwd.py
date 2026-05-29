# LUCID Backward kernel — computes dV and dK for forward substitution.
#
# Phase 1 (upper triangle, sequential with flag sync):
#   dV_i = inv(L_ii^T) @ (dV'_i - sum_{j>i} exp(K_i@K_j^T) @ dV_j)
#   acc_dK += upper triangle contributions from dS^T @ K_j
#
# Phase 2 (lower triangle, parallel, no sync):
#   acc_dK += lower triangle contributions from dS^T @ K_j
#
# Per-CTA (one per block i, reverse scheduled):
#   Phase 1: GEMM1(S) + GEMM2(G) + PW(dS) + GEMM3(dV) + GEMM5(dK) — 4-5 GEMMs
#   Diagonal solve → dV TMA write → flag signal → sVfixed reload
#   Phase 2: GEMM1(S) + GEMM2(G) + PW(dS) + GEMM5(dK) — 3-4 GEMMs
#   dK TMA write

import math
from types import SimpleNamespace
from typing import Type, Callable, Optional, Tuple
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic

from quack import layout_utils
from quack import sm90_utils
from quack import copy_utils

from lucid_kernels.cute.cute_dsl_utils import assume_tensor_aligned
from lucid_kernels.cute import utils
from lucid_kernels.cute import pipeline
from lucid_kernels.cute.named_barrier import NamedBarrierFwd
from quack.cute_dsl_utils import ParamsBase
from lucid_kernels.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
)
from lucid_kernels.cute.clock_utils import read_globaltimer, store_ts

# DiagSync barrier — after NamedBarrierFwd's PEmpty=6
DIAG_SYNC_BARRIER_ID = 7


class BackwardSubSm90:
    """LUCID backward kernel — computes dV (sequential) and optionally dK (parallel).

    Phase 1: dV_i = inv(L_ii^T) @ (dV'_i - sum_{j>i} exp(K_i@K_j^T) @ dV_j)
             acc_dK += sum_{j>i} dS_ij^T @ K_j  (upper triangle)
    Phase 2: acc_dK += sum_{j<i} dS_ij^T @ K_j  (lower triangle)
    """

    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_stages: int = 2,
        num_threads: int = 384,
        window_size: Optional[int] = None,  # None = full causality, int = last W blocks
        no_sync: bool = False,  # Skip flag synchronization (for profiling)
        instrument: bool = False,  # Record per-CTA globaltimer timestamps
        use_diag_solve: bool = False,  # Apply diagonal block solve after off-diag corrections
        compute_dk: bool = False,  # Also compute dK (Phase 1 upper + Phase 2 lower triangle)
    ):
        self.dtype = dtype
        hdim_multiple_of = 16
        self.head_dim = head_dim
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.head_dim_v = head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.tile_m = m_block_size
        self.tile_n = n_block_size
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.mma_pv_is_rs = True
        self.window_size = window_size
        self.no_sync = no_sync
        self.instrument = instrument
        self.use_diag_solve = use_diag_solve
        self.compute_dk = compute_dk
        self.buffer_align_bytes = 1024

    def _get_tiled_mma(self):
        # QK GEMM: K_i @ K_j^T → (m_block, n_block). Both operands K-major.
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        # PV GEMM: P @ V' → (m_block, head_dim_v). B operand (V') is MN-major.
        # a_source=RMEM because P is in registers (mma_pv_is_rs=True).
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM,
        )
        tiled_mma_pv_rs = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM,
        )
        # Diagonal solve: diag_inv(SMEM, K-major) @ RHS^T(SMEM, MN-major) → acc_O
        tiled_mma_diag = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,    # A: diag_inv in sV_recv (K-major like sQ)
            warpgroup.OperandMajorMode.MN,   # B: RHS transposed in sOt (MN-major like sVt)
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_hdimv),
        )
        # dK GEMM5: dK(n_block, hdim) += dS^T(n_block, m_block) @ K_j(m_block, hdim)
        # A = dS^T from registers (RS MMA), B = K_j^T from sKt (MN-major)
        tiled_mma_dK = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,    # A: dS^T from registers
            warpgroup.OperandMajorMode.MN,   # B: K_j^T from transposed sK
            Float32,
            atom_layout_mnk=(self.tile_n // 64, 1, 1),
            tiler_mn=(64, self.tile_hdim),
            a_source=warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_rs, tiled_mma_diag, tiled_mma_dK

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, 0], 1024]
        mbar_ptr_QO_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        # sV_recv buffer for diag_inv (same size as sO)
        sV_recv_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sO_layout)], 128
        ]
        # 1 mbarrier for diag_inv TMA load
        mbar_diag_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 128]
        # 1 mbarrier for V'_i fixed load (compute_dk path)
        mbar_vfixed_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 128]
        # sVfixed: fixed buffer for V'_i (Phase 1) / dV_i (Phase 2) — only when compute_dk
        sVfixed_cosize = cute.cosize(self.sVfixed_layout) if self.compute_dk else 0
        sVfixed_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, sVfixed_cosize], 128]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr: mbar_ptr_QO_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_diag: mbar_diag_struct
            mbar_vfixed: mbar_vfixed_struct
            sV_recv: sV_recv_struct
            sVfixed: sVfixed_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        return SharedStorageQKV

    # =========================================================================
    # __call__: entry point — sets up TMA descriptors and launches kernel
    # =========================================================================
    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,       # (batch, seqlen, heads_kv, head_dim)
        mV: cute.Tensor,       # (batch, seqlen, heads_kv, head_dim_v)
        mVprime: cute.Tensor,  # (batch, seqlen, heads_kv, head_dim_v) — output & streaming input
        mFlags: cute.Tensor,   # (heads_kv * batch, T) int32 — sync flags
        mVorig: cute.Tensor,   # (batch, seqlen, heads_kv, head_dim_v) — dV' upstream gradient
        mTimestamps: cute.Tensor,  # (T, 8) int64 — debug timestamps
        mDiagInv: cute.Tensor, # (batch, seqlen, heads_kv, block_size) — diag_inv
        stream: cuda.CUstream,
        mVprimeFwd: Optional[cute.Tensor] = None,  # V' from forward pass (Phase 2 streaming + fixed V'_i)
        mdK: Optional[cute.Tensor] = None,         # dK output (seqlen, head_dim, num_heads_kv, batch)
    ):
        # Assume alignment and transpose BSHD → (S, D, H, B)
        mK, mV, mVprime, mVorig, mDiagInv = [
            assume_tensor_aligned(t) for t in (mK, mV, mVprime, mVorig, mDiagInv)
        ]
        layout_transpose = [1, 3, 2, 0]
        mK, mV, mVprime, mVorig, mDiagInv = [
            layout_utils.select(t, layout_transpose)
            for t in (mK, mV, mVprime, mVorig, mDiagInv)
        ]

        # When compute_dk: process mVprimeFwd (V' from fwd pass) and mdK (dK output)
        # When not compute_dk: use existing tensors as dummies (never accessed)
        if const_expr(self.compute_dk):
            mVprimeFwd, mdK = [
                assume_tensor_aligned(t) for t in (mVprimeFwd, mdK)
            ]
            mVprimeFwd, mdK = [
                layout_utils.select(t, layout_transpose)
                for t in (mVprimeFwd, mdK)
            ]
        else:
            mVprimeFwd = mVprime  # dummy, never used
            mdK = mK  # dummy, never used

        tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_rs, tiled_mma_diag, tiled_mma_dK = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_mma_warp_groups = self.num_mma_threads // self.num_threads_per_warp_group
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_mma_threads
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs = 240
        self.num_producer_regs = 24
        self.use_scheduler_barrier = False
        self.use_tma_O = True

        # Create smem layouts inline using upstream pattern
        self.sQ_layout = sm90_utils.make_smem_layout(
            mK.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), None
        )
        self.sK_layout = sm90_utils.make_smem_layout(
            mK.element_type, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), self.num_stages
        )
        self.sV_layout = sm90_utils.make_smem_layout(
            mVprime.element_type, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), self.num_stages
        )
        self.sO_layout = sm90_utils.make_smem_layout(
            mVprime.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdimv), None
        )
        self.sV_recv_layout = self.sO_layout
        self.sVfixed_layout = sm90_utils.make_smem_layout(
            mVprime.element_type, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), None
        )

        SharedStorage = self._get_shared_storage_cls()

        # TMA descriptors
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()

        self.tma_copy_bytes = {
            "Q": cute.size_in_bytes(mK.element_type, cute.select(self.sQ_layout, mode=[0, 1])),
            "K": cute.size_in_bytes(mK.element_type, cute.select(self.sK_layout, mode=[0, 1])),
            "V": cute.size_in_bytes(mVprime.element_type, cute.select(self.sV_layout, mode=[0, 1])),
        }
        self.tma_copy_vfixed_bytes = cute.size_in_bytes(mVprimeFwd.element_type, self.sVfixed_layout)

        # Q-slot TMA → loads K_i as the "query"
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q, mK, self.sQ_layout, (self.tile_m, self.tile_hdim),
        )
        # K-slot TMA → loads K_j for streaming
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV, mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim), 1,
        )
        # V-slot TMA → loads dV_j from dV buffer for streaming
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV, mVprime,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv), 1,
        )
        # O-slot TMA → writes dV_i to dV buffer
        tma_atom_O, tma_tensor_Vprime = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_O, mVprime, self.sO_layout, (self.tile_m, self.tile_hdimv),
        )
        # D-slot TMA → loads diag_inv_i into sV_recv (same layout as Q-slot: BS×BS K-major)
        tma_atom_D, tma_tensor_D = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q, mDiagInv, self.sV_recv_layout,
            (self.tile_m, self.tile_hdimv),
        )
        # Vfixed-slot TMA → loads V'_i into sVfixed (G2S, single tile, no staging)
        tma_atom_Vfixed, tma_tensor_Vfixed = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q, mVprimeFwd,
            self.sVfixed_layout,
            (self.tile_n, self.tile_hdimv),
        )
        # Vprime-slot TMA → streams V'_j in Phase 2 (G2S, uses sV pipeline layout)
        tma_atom_Vprime, tma_tensor_Vprime2 = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV, mVprimeFwd,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv), 1,
        )
        # dK-slot TMA → writes dK from sO to mdK (S2G)
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_O, mdK, self.sO_layout,
            (self.tile_m, self.tile_hdim),
        )

        # Tile scheduler: one block per (m_block, head, batch)
        T_blocks = cute.ceil_div(cute.size(mK.shape[0]), self.tile_m)
        tile_sched_args = TileSchedulerArguments(
            T_blocks,
            cute.size(mK.shape[2]),
            cute.size(mK.shape[3]),
            1,  # num_splits
            cute.size(mK.shape[0]),
            mK.shape[1],
            mVprime.shape[1],
            total_q=cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=False,
        )
        TileScheduler = SingleTileScheduler
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_Vprime,
            mVorig,
            mFlags,
            mTimestamps,
            tma_tensor_D,
            tma_tensor_Vfixed,
            tma_tensor_Vprime2,
            tma_tensor_dK,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            tma_atom_D,
            tma_atom_Vfixed,
            tma_atom_Vprime,
            tma_atom_dK,
            Float32(LOG2_E),
            Int32(T_blocks),
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sV_recv_layout,
            self.sVfixed_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_pv_rs,
            tiled_mma_diag,
            tiled_mma_dK,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # =========================================================================
    # kernel: CUDA kernel entry — splits into producer/consumer warp groups
    # Block mapping reversed: CTA m_block → physical block (T-1-m_block)
    # =========================================================================
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,       # K data (Q-slot TMA tensor)
        mK: cute.Tensor,       # K data (K-slot TMA tensor)
        mV: cute.Tensor,       # dV data (V-slot TMA tensor, streaming solved dV_j)
        mO: cute.Tensor,       # dV data (O-slot TMA tensor, output)
        mVorig: cute.Tensor,   # dV' (upstream gradient)
        mFlags: cute.Tensor,   # Flags (num_heads_kv * batch, T)
        mTimestamps: cute.Tensor,
        mDiagInv: cute.Tensor,
        mVfixed: cute.Tensor,  # V' from fwd (Vfixed-slot TMA tensor)
        mVprime2: cute.Tensor, # V' from fwd (Vprime-slot TMA tensor, Phase 2 streaming)
        mdK: cute.Tensor,      # dK output (dK-slot TMA tensor)
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        tma_atom_D: cute.CopyAtom,
        tma_atom_Vfixed: cute.CopyAtom,
        tma_atom_Vprime: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        log2e: Float32,
        T_blocks: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sV_recv_layout: cute.ComposedLayout,
        sVfixed_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        tiled_mma_diag: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch TMA descriptors
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_O)
            if const_expr(self.use_diag_solve):
                cpasync.prefetch_descriptor(tma_atom_D)
            if const_expr(self.compute_dk):
                cpasync.prefetch_descriptor(tma_atom_Vfixed)
                cpasync.prefetch_descriptor(tma_atom_Vprime)
                cpasync.prefetch_descriptor(tma_atom_dK)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier init
        mbar_ptr_Q = storage.mbar_ptr.data_ptr()
        if warp_idx == 1:
            cute.arch.mbarrier_init(mbar_ptr_Q, 1)

        # Diag solve mbarrier init
        mbar_diag_ptr = storage.mbar_diag.data_ptr()
        if const_expr(self.use_diag_solve):
            if warp_idx == 1:
                cute.arch.mbarrier_init(mbar_diag_ptr, 1)

        # Vfixed mbarrier init (for V'_i load into sVfixed)
        mbar_vfixed_ptr = storage.mbar_vfixed.data_ptr()
        if const_expr(self.compute_dk):
            if warp_idx == 1:
                cute.arch.mbarrier_init(mbar_vfixed_ptr, 1)

        pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["K"],
            defer_sync=True,
        )
        pipeline_v = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["V"],
        )

        # Get shared memory buffers
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = layout_utils.transpose_view(sV)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

        # sV_recv buffer for diag_inv
        sV_recv = storage.sV_recv.get_tensor(
            sV_recv_layout.outer, swizzle=sV_recv_layout.inner, dtype=self.dtype
        )

        # sVfixed: fixed buffer for V'_i (Phase 1) / dV_i (Phase 2)
        if const_expr(self.compute_dk):
            sVfixed = storage.sVfixed.get_tensor(
                sVfixed_layout.outer, swizzle=sVfixed_layout.inner, dtype=self.dtype
            )
        else:
            sVfixed = sQ  # dummy, never used

        # Create tile scheduler callable BEFORE the producer/consumer split
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Warp specialization: producer (warp 0-3) vs consumer (warp 4+)
        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ, mK, mV, sQ, sK, sV,
                tma_atom_Q, tma_atom_K, tma_atom_V,
                pipeline_k, pipeline_v, mbar_ptr_Q,
                mFlags, mTimestamps,
                TileSchedulerCls,
                tma_atom_D, mDiagInv, sV_recv, mbar_diag_ptr,
                T_blocks,
                # compute_dk args
                mVfixed, tma_atom_Vfixed, sVfixed, mbar_vfixed_ptr,
                mVprime2, tma_atom_Vprime,
            )
        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk, tiled_mma_pv, tiled_mma_pv_rs, tiled_mma_diag,
                tiled_mma_dK,
                mO, mVorig, mFlags,
                sQ, sK, sVt, sO,
                pipeline_k, pipeline_v, mbar_ptr_Q,
                tma_atom_O, mTimestamps,
                tidx, log2e,
                TileSchedulerCls,
                sV_recv,
                mbar_diag_ptr,
                T_blocks,
                # compute_dk args
                sVfixed, mbar_vfixed_ptr,
                mdK, tma_atom_dK,
                mQ, tma_atom_Q,
            )

    # =========================================================================
    # load: Producer — loads K_i (Q-slot) and streams K_j, dV_j via TMA
    # Block mapping reversed: CTA m_block → physical block (T-1-m_block)
    # Inner loop: j > i (upper triangle), farthest first
    # =========================================================================
    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        mFlags: cute.Tensor,
        mTimestamps: cute.Tensor,
        TileSchedulerCls: Callable,
        tma_atom_D: cute.CopyAtom,
        mDiagInv: cute.Tensor,
        sV_recv: cute.Tensor,
        mbar_diag_ptr: cutlass.Pointer,
        T_blocks: Int32,
        # compute_dk args
        mVfixed: cute.Tensor,
        tma_atom_Vfixed: cute.CopyAtom,
        sVfixed: cute.Tensor,
        mbar_vfixed_ptr: cutlass.Pointer,
        mVprime2: cute.Tensor,
        tma_atom_Vprime: cute.CopyAtom,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            kv_producer_state = pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                m_block_sched, head_idx, batch_idx, _ = work_tile.tile_idx
                # Reverse mapping: CTA 0 → block T-1, CTA 1 → block T-2, ...
                m_block = T_blocks - Int32(1) - m_block_sched

                mQ_cur = mQ[None, None, head_idx, batch_idx]
                mK_cur = mK[None, None, head_idx, batch_idx]
                mV_cur = mV[None, None, head_idx, batch_idx]

                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))

                tQsQ, tQgQ = cpasync.tma_partition(
                    tma_atom_Q, 0, cute.make_layout(1),
                    cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2),
                )
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K, 0, cute.make_layout(1),
                    cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V, 0, cute.make_layout(1),
                    cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2),
                )

                load_K = partial(self.load_K, tma_atom_K, tKgK, tKsK, pipeline_k)
                load_V = partial(self.load_K, tma_atom_V, tVgV, tVsV, pipeline_v)

                # Load K_i into Q-slot smem
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_ptr_Q, self.tma_copy_bytes["Q"]
                    )
                cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=mbar_ptr_Q)

                # Preload diag_inv_transpose_i into sV_recv early
                if const_expr(self.use_diag_solve):
                    mDiagInv_cur = mDiagInv[None, None, head_idx, batch_idx]
                    gDiag = cute.local_tile(
                        mDiagInv_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
                    )
                    tDsD, tDgD = cpasync.tma_partition(
                        tma_atom_D, 0, cute.make_layout(1),
                        cute.group_modes(sV_recv, 0, 2),
                        cute.group_modes(gDiag, 0, 2),
                    )
                    diag_copy_bytes = (
                        self.tile_m * self.tile_hdimv * (self.dtype.width // 8)
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_diag_ptr, diag_copy_bytes
                        )
                    cute.copy(tma_atom_D, tDgD, tDsD, tma_bar_ptr=mbar_diag_ptr)

                # Preload V'_i into sVfixed for GEMM2 (dK path)
                if const_expr(self.compute_dk):
                    mVf_cur = mVfixed[None, None, head_idx, batch_idx]
                    gVf = cute.local_tile(
                        mVf_cur, (self.tile_n, self.tile_hdimv), (m_block, 0)
                    )
                    tVfsVf, tVfgVf = cpasync.tma_partition(
                        tma_atom_Vfixed, 0, cute.make_layout(1),
                        cute.group_modes(sVfixed, 0, 2),
                        cute.group_modes(gVf, 0, 2),
                    )
                    vfixed_copy_bytes = (
                        self.tile_n * self.tile_hdimv * (self.dtype.width // 8)
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_vfixed_ptr, vfixed_copy_bytes
                        )
                    cute.copy(
                        tma_atom_Vfixed, tVfgVf, tVfsVf,
                        tma_bar_ptr=mbar_vfixed_ptr,
                    )

                # --- Instrument: producer before inner loop ---
                if const_expr(self.instrument):
                    with cute.arch.elect_one():
                        gTs0 = cute.local_tile(mTimestamps, (1, 1), (m_block, 0))
                        store_ts(gTs0.iterator, read_globaltimer())

                # For block i, load K_j and dV_j for j > i (upper triangle)
                # Farthest first: j = T-1, T-2, ..., i+1
                n_block_count = (T_blocks - Int32(1)) - m_block
                for j_iter in cutlass.range(n_block_count, unroll=2):
                    n_block = (T_blocks - Int32(1)) - j_iter
                    load_K(n_block, producer_state=kv_producer_state)
                    if const_expr(not self.no_sync):
                        head_batch_idx = (
                            head_idx * cute.size(mK.shape[3]) + batch_idx
                        )
                        gFlag = cute.local_tile(
                            mFlags, (1, 1), (head_batch_idx, n_block)
                        )
                        flag_ptr = gFlag.iterator
                        with cute.arch.elect_one():
                            flag_val = cute.arch.atomic_add(
                                flag_ptr, Int32(0),
                                sem="acquire", scope="gpu",
                            )
                            while flag_val == 0:
                                flag_val = cute.arch.atomic_add(
                                    flag_ptr, Int32(0),
                                    sem="acquire", scope="gpu",
                                )
                    load_V(n_block, producer_state=kv_producer_state)
                    kv_producer_state.advance()

                # --- Instrument: producer after Phase 1 inner loop ---
                if const_expr(self.instrument):
                    with cute.arch.elect_one():
                        gTs1 = cute.local_tile(mTimestamps, (1, 1), (m_block, 1))
                        store_ts(gTs1.iterator, read_globaltimer())

                # Phase 2: stream K_j and V'_j for j < i (ascending, lower triangle)
                if const_expr(self.compute_dk):
                    phase2_count = m_block
                    mVp2_cur = mVprime2[None, None, head_idx, batch_idx]
                    gVp2 = cute.local_tile(
                        mVp2_cur, (self.tile_n, self.tile_hdimv), (None, 0)
                    )
                    tVp2sV, tVp2gV = cpasync.tma_partition(
                        tma_atom_Vprime, 0, cute.make_layout(1),
                        cute.group_modes(sV, 0, 2),
                        cute.group_modes(gVp2, 0, 2),
                    )
                    load_Vprime = partial(
                        self.load_K, tma_atom_Vprime, tVp2gV, tVp2sV, pipeline_v
                    )
                    for j in cutlass.range(phase2_count, unroll=2):
                        load_K(j, producer_state=kv_producer_state)
                        load_Vprime(j, producer_state=kv_producer_state)
                        kv_producer_state.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    def load_K(
        self,
        tma_atom: cute.CopyAtom,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        pipeline: cutlass.pipeline.PipelineAsync,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
    ):
        pipeline.producer_acquire(producer_state)
        cute.copy(
            tma_atom,
            tKgK[None, block],
            tKsK[None, producer_state.index],
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
        )

    # =========================================================================
    # mma: Consumer — accumulates corrections and writes dV_i
    # =========================================================================
    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        tiled_mma_diag: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        mO: cute.Tensor,
        mVorig: cute.Tensor,
        mFlags: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sO: cute.Tensor,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        tma_atom_O: cute.CopyAtom,
        mTimestamps: cute.Tensor,
        tidx: Int32,
        log2e: Float32,
        TileSchedulerCls: Callable,
        sV_recv: cute.Tensor,
        mbar_diag_ptr: cutlass.Pointer,
        T_blocks: Int32,
        # compute_dk args
        sVfixed: cute.Tensor,
        mbar_vfixed_ptr: cutlass.Pointer,
        mdK: cute.Tensor,
        tma_atom_dK: cute.CopyAtom,
        mQ: cute.Tensor = None,
        tma_atom_Q: cute.CopyAtom = None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )

        # QK GEMM fragments — use upstream partition_fragment_ABC
        wg_mma_qk = tiled_mma_qk.get_slice(
            warp_group_thread_layout(warp_group_idx)
        )
        wg_mma_pv = tiled_mma_pv.get_slice(
            warp_group_thread_layout(warp_group_idx)
        )
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        # PV GEMM fragments (sP=None since mma_pv_is_rs=True)
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), None, sVt
        )

        # Diagonal solve fragments
        if const_expr(self.use_diag_solve):
            wg_mma_diag = tiled_mma_diag.get_slice(
                warp_group_thread_layout(warp_group_idx)
            )
            tDrDiag = tiled_mma_diag.make_fragment_A(
                wg_mma_diag.partition_A(sV_recv)
            )
            sOt = layout_utils.transpose_view(sO)
            tDrRHSt = tiled_mma_diag.make_fragment_B(
                wg_mma_diag.partition_B(sOt)
            )

        # Number of rows for the exp loop
        acc_S_shape = tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        num_rows = acc_S_shape[0][0] * acc_S_shape[1]

        mma_params = SimpleNamespace(
            tSrQ=tSrQ, tSrK=tSrK, tOrP=tOrP, tOrVt=tOrVt, acc_O=acc_O
        )

        # dK computation infrastructure
        if const_expr(self.compute_dk):
            # Recover untransposed sV from sVt for GEMM2 B operand (K-major)
            sV_untransp = layout_utils.transpose_view(sVt)
            # Transposed sK for GEMM5 B operand (MN-major)
            sKt = layout_utils.transpose_view(sK)

            # GEMM2: acc_G = V'_i(sVfixed) @ dV_j(sV)^T — reuses tiled_mma_qk
            tGrVfixed = tiled_mma_qk.make_fragment_A(
                wg_mma_qk.partition_A(sVfixed)
            )
            tGrV = tiled_mma_qk.make_fragment_B(
                wg_mma_qk.partition_B(sV_untransp)
            )

            # GEMM5: acc_dK += dS_frgA @ K_j^T(sKt) — RS MMA via tiled_mma_dK
            wg_mma_dK = tiled_mma_dK.get_slice(
                warp_group_thread_layout(warp_group_idx)
            )
            tGrKt = tiled_mma_dK.make_fragment_B(wg_mma_dK.partition_B(sKt))

            # dK accumulator — zero-initialized per block
            acc_shape_dK = tiled_mma_dK.partition_shape_C(
                (self.tile_m, self.tile_hdim)
            )
            acc_dK = cute.make_fragment(acc_shape_dK, Float32)

            mma_params.tGrVfixed = tGrVfixed
            mma_params.tGrV = tGrV
            mma_params.tGrKt = tGrKt
            mma_params.acc_dK = acc_dK

        q_consumer_phase = Int32(0)
        diag_consumer_phase = Int32(0)
        vfixed_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block_sched, head_idx, batch_idx, _ = work_tile.tile_idx
            # Reverse mapping: CTA 0 → block T-1, CTA 1 → block T-2, ...
            m_block = T_blocks - Int32(1) - m_block_sched

            # Wait for K_i (Q-slot) to be loaded
            cute.arch.mbarrier_wait(mbar_ptr_Q, phase=q_consumer_phase)
            q_consumer_phase ^= 1

            # --- Instrument: consumer after K_i loaded ---
            if const_expr(self.instrument):
                with cute.arch.elect_one():
                    gTs2 = cute.local_tile(mTimestamps, (1, 1), (m_block, 2))
                    store_ts(gTs2.iterator, read_globaltimer())

            # Initialize acc_O = dV'_i (upstream gradient, prefetch from HBM).
            mVorig_cur = mVorig[None, None, head_idx, batch_idx]
            gVi = cute.local_tile(
                mVorig_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
            )
            thr_mma_pv = tiled_mma_pv.get_slice(tidx)
            taccOgVi = thr_mma_pv.partition_C(gVi)
            acc_O.store(taccOgVi.load().to(Float32))

            # Wait for V'_i in sVfixed and zero acc_dK
            if const_expr(self.compute_dk):
                cute.arch.mbarrier_wait(
                    mbar_vfixed_ptr, phase=vfixed_consumer_phase
                )
                vfixed_consumer_phase ^= 1
                for _zi in cutlass.range(cute.size(acc_dK)):
                    acc_dK[_zi] = Float32(0.0)

            # Upper triangle: j > i, n_block_count = (T-1) - i
            n_block_count = (T_blocks - Int32(1)) - m_block

            # Accumulate corrections (matches producer's farthest-first order)
            for j_iter in cutlass.range(n_block_count, unroll=1):
                n_block = (T_blocks - Int32(1)) - j_iter
                if const_expr(self.compute_dk):
                    kv_consumer_state = self.sub_one_n_block_dk(
                        n_block, kv_consumer_state,
                        tiled_mma_qk, tiled_mma_pv, tiled_mma_dK,
                        pipeline_k, pipeline_v,
                        mma_params, log2e, num_rows,
                        O_should_accumulate=True,
                    )
                else:
                    kv_consumer_state = self.sub_one_n_block(
                        n_block, kv_consumer_state,
                        tiled_mma_qk, tiled_mma_pv,
                        pipeline_k, pipeline_v,
                        mma_params, log2e, num_rows,
                        O_should_accumulate=True,
                    )

            # --- Instrument: consumer after MMA loop ---
            if const_expr(self.instrument):
                with cute.arch.elect_one():
                    gTs3 = cute.local_tile(mTimestamps, (1, 1), (m_block, 3))
                    store_ts(gTs3.iterator, read_globaltimer())

            # Diagonal solve: dV_i = diag_inv_i @ RHS_i
            if const_expr(self.use_diag_solve):
                # 1. Store RHS (acc_O) → sO as bf16/fp16
                rO_diag = cute.make_fragment_like(acc_O, self.dtype)
                rO_diag.store(acc_O.load().to(self.dtype))
                smem_copy_atom_diag = copy_utils.get_smem_store_atom(
                    self.arch, self.dtype
                )
                smem_thr_copy_diag = cute.make_tiled_copy_C(
                    smem_copy_atom_diag, tiled_mma_pv
                ).get_slice(tidx)
                taccOrO_diag = smem_thr_copy_diag.retile(rO_diag)
                taccOsO_diag = smem_thr_copy_diag.partition_D(sO)
                cute.copy(smem_copy_atom_diag, taccOrO_diag, taccOsO_diag)

                # 2. Fence + barrier sync
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                cute.arch.barrier(
                    barrier_id=DIAG_SYNC_BARRIER_ID,
                    number_of_threads=self.num_epilogue_threads,
                )

                # 3. Wait for diag_inv TMA load into sV_recv
                cute.arch.mbarrier_wait(
                    mbar_diag_ptr, phase=diag_consumer_phase
                )
                diag_consumer_phase ^= 1

                # 4. WGMMA: acc_O = diag_inv @ RHS^T
                sm90_utils.gemm(
                    tiled_mma_diag, acc_O, tDrDiag, tDrRHSt,
                    zero_init=True, wg_wait=0,
                )

            # --- Instrument: after diag_solve, before epilogue ---
            if const_expr(self.instrument):
                with cute.arch.elect_one():
                    gTs6 = cute.local_tile(mTimestamps, (1, 1), (m_block, 6))
                    store_ts(gTs6.iterator, read_globaltimer())

            # Epilogue: dV_i write + flag signal
            self.forward_sub_epilogue(
                acc_O, mO, mVorig, mFlags,
                sO, tma_atom_O, mTimestamps,
                tiled_mma_pv, tidx,
                m_block, head_idx, batch_idx,
            )

            # === Phase 2: lower triangle dK accumulation (j < i) ===
            if const_expr(self.compute_dk):
                # Reload K_i into sQ via TMA (epilogue destroyed it via sO/sQ alias)
                mQ_cur = mQ[None, None, head_idx, batch_idx]
                gQ_reload = cute.local_tile(
                    mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0)
                )
                tQsQ_r, tQgQ_r = cpasync.tma_partition(
                    tma_atom_Q, 0, cute.make_layout(1),
                    cute.group_modes(sQ, 0, 2),
                    cute.group_modes(gQ_reload, 0, 2),
                )
                warp_idx_reload = cute.arch.make_warp_uniform(
                    cute.arch.warp_idx()
                )
                if warp_idx_reload == 4:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_ptr_Q, self.tma_copy_bytes["Q"]
                        )
                    cute.copy(
                        tma_atom_Q, tQgQ_r, tQsQ_r,
                        tma_bar_ptr=mbar_ptr_Q,
                    )
                cute.arch.mbarrier_wait(
                    mbar_ptr_Q, phase=q_consumer_phase
                )
                q_consumer_phase ^= 1

                # Reload dV_i from acc_O registers into sVfixed
                rO_reload = cute.make_fragment_like(acc_O, self.dtype)
                rO_reload.store(acc_O.load().to(self.dtype))
                smem_copy_atom_vf = copy_utils.get_smem_store_atom(
                    self.arch, self.dtype
                )
                smem_thr_copy_vf = cute.make_tiled_copy_C(
                    smem_copy_atom_vf, tiled_mma_pv
                ).get_slice(tidx)
                taccOrO_vf = smem_thr_copy_vf.retile(rO_reload)
                taccOsVf = smem_thr_copy_vf.partition_D(sVfixed)
                cute.copy(smem_copy_atom_vf, taccOrO_vf, taccOsVf)
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                cute.arch.barrier(
                    barrier_id=DIAG_SYNC_BARRIER_ID,
                    number_of_threads=self.num_epilogue_threads,
                )

                # Phase 2 loop: j = 0 to i-1 (ascending)
                phase2_count = m_block
                for j in cutlass.range(phase2_count, unroll=1):
                    kv_consumer_state = self.sub_one_n_block_dk_p2(
                        j, kv_consumer_state,
                        tiled_mma_qk, tiled_mma_dK,
                        pipeline_k, pipeline_v,
                        mma_params, log2e, num_rows,
                    )

                # dK epilogue: write acc_dK to HBM via TMA
                self.dk_epilogue(
                    mma_params.acc_dK, mdK, sO, tma_atom_dK,
                    tiled_mma_dK, tidx,
                    m_block, head_idx, batch_idx,
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # =========================================================================
    # sub_one_n_block: Process one off-diagonal block (dV only, no dK)
    # =========================================================================
    @cute.jit
    def sub_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mma_params: SimpleNamespace,
        log2e: Float32,
        num_rows: cutlass.Constexpr,
        O_should_accumulate: cutlass.Boolean = True,
    ):
        acc_S = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )

        # Wait for K_j, compute S = K_i @ K_j^T
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        sm90_utils.gemm(
            tiled_mma_qk, acc_S, mma_params.tSrQ,
            mma_params.tSrK[None, None, None, smem_pipe_read.index],
            zero_init=True, wg_wait=-1,
        )
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        # Element-wise -exp: P = -exp(S) = -exp2(S * log2(e))
        acc_S_mn = layout_utils.make_acc_tensor_mn_view(acc_S)
        for r in cutlass.range(num_rows, unroll_full=True):
            row = acc_S_mn[r, None].load()
            acc_S_mn[r, None].store(
                -cute.math.exp2(row * log2e, fastmath=True)
            )

        # Convert to fp16/bf16
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP = mma_params.tOrP
        utils.cvt_f16(tOrP_acc, tOrP)

        # Wait for dV_j, compute acc_O += P @ dV_j
        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        sm90_utils.gemm(
            tiled_mma_pv, mma_params.acc_O, mma_params.tOrP,
            mma_params.tOrVt[None, None, None, smem_pipe_read.index],
            zero_init=not O_should_accumulate, wg_wait=0,
        )
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    # =========================================================================
    # sub_one_n_block_dk: Process one off-diagonal block with dK accumulation
    # Performs 5 GEMMs: S, G, P@V(dV), dS@K(dK), plus pointwise
    # =========================================================================
    @cute.jit
    def sub_one_n_block_dk(
        self,
        n_block: Int32,
        smem_pipe_read: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mma_params: SimpleNamespace,
        log2e: Float32,
        num_rows: cutlass.Constexpr,
        O_should_accumulate: cutlass.Boolean = True,
    ):
        # Allocate accumulators
        acc_S = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )
        acc_G = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )

        # GEMM1: S = K_i(sQ) @ K_j(sK)^T
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        sm90_utils.gemm(
            tiled_mma_qk, acc_S, mma_params.tSrQ,
            mma_params.tSrK[None, None, None, smem_pipe_read.index],
            zero_init=True, wg_wait=-1,
        )

        # GEMM2: G = V'_i(sVfixed) @ dV_j(sV)^T
        warpgroup.wait_group(0)
        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        sm90_utils.gemm(
            tiled_mma_qk, acc_G, mma_params.tGrVfixed,
            mma_params.tGrV[None, None, None, smem_pipe_read.index],
            zero_init=True, wg_wait=-1,
        )
        warpgroup.wait_group(0)

        # Pointwise: P = -exp(S), dS = -G * exp(S)
        acc_S_mn = layout_utils.make_acc_tensor_mn_view(acc_S)
        acc_G_mn = layout_utils.make_acc_tensor_mn_view(acc_G)
        for r in cutlass.range(num_rows, unroll_full=True):
            s_row = acc_S_mn[r, None].load()
            g_row = acc_G_mn[r, None].load()
            exp_s = cute.math.exp2(s_row * log2e, fastmath=True)
            acc_G_mn[r, None].store(-g_row * exp_s)  # dS = -G * exp_S
            acc_S_mn[r, None].store(-exp_s)           # P = -exp_S

        # Convert P → tOrP (for GEMM3)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        utils.cvt_f16(tOrP_acc, mma_params.tOrP)

        # Convert dS → tOrP_dS (for GEMM5)
        tOrP_dS_acc = layout_utils.reshape_acc_to_frgA(acc_G)
        tOrP_dS = cute.make_fragment_like(tOrP_dS_acc, self.dtype)
        utils.cvt_f16(tOrP_dS_acc, tOrP_dS)

        # GEMM3: acc_O += P @ dV_j(sVt) — dV accumulation
        sm90_utils.gemm(
            tiled_mma_pv, mma_params.acc_O, mma_params.tOrP,
            mma_params.tOrVt[None, None, None, smem_pipe_read.index],
            zero_init=not O_should_accumulate, wg_wait=-1,
        )

        # GEMM5: acc_dK += dS @ K_j^T(sKt) — dK accumulation (RS MMA)
        sm90_utils.gemm(
            tiled_mma_dK, mma_params.acc_dK, tOrP_dS,
            mma_params.tGrKt[None, None, None, smem_pipe_read.index],
            zero_init=False, wg_wait=0,
        )

        # Release both pipelines
        pipeline_k.consumer_release(smem_pipe_read)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    # =========================================================================
    # sub_one_n_block_dk_p2: Phase 2 — lower triangle dK only (no dV update)
    # GEMM1(S) + GEMM2(G=dV_i@V'_j^T) + PW(dS) + GEMM5(dK)
    # =========================================================================
    @cute.jit
    def sub_one_n_block_dk_p2(
        self,
        n_block: Int32,
        smem_pipe_read: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mma_params: SimpleNamespace,
        log2e: Float32,
        num_rows: cutlass.Constexpr,
    ):
        # Allocate accumulators
        acc_S = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )
        acc_G = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )

        # GEMM1: S = K_i(sQ) @ K_j(sK)^T
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        sm90_utils.gemm(
            tiled_mma_qk, acc_S, mma_params.tSrQ,
            mma_params.tSrK[None, None, None, smem_pipe_read.index],
            zero_init=True, wg_wait=-1,
        )

        # GEMM2: G = dV_i(sVfixed) @ V'_j(sV)^T — reuses tiled_mma_qk
        warpgroup.wait_group(0)
        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        sm90_utils.gemm(
            tiled_mma_qk, acc_G, mma_params.tGrVfixed,
            mma_params.tGrV[None, None, None, smem_pipe_read.index],
            zero_init=True, wg_wait=-1,
        )
        warpgroup.wait_group(0)

        # Pointwise: dS = -G * exp(S)
        acc_S_mn = layout_utils.make_acc_tensor_mn_view(acc_S)
        acc_G_mn = layout_utils.make_acc_tensor_mn_view(acc_G)
        for r in cutlass.range(num_rows, unroll_full=True):
            s_row = acc_S_mn[r, None].load()
            g_row = acc_G_mn[r, None].load()
            exp_s = cute.math.exp2(s_row * log2e, fastmath=True)
            acc_G_mn[r, None].store(-g_row * exp_s)

        # Convert dS → tOrP_dS (for GEMM5)
        tOrP_dS_acc = layout_utils.reshape_acc_to_frgA(acc_G)
        tOrP_dS = cute.make_fragment_like(tOrP_dS_acc, self.dtype)
        utils.cvt_f16(tOrP_dS_acc, tOrP_dS)

        # GEMM5: acc_dK += dS @ K_j^T(sKt) — dK accumulation (RS MMA)
        sm90_utils.gemm(
            tiled_mma_dK, mma_params.acc_dK, tOrP_dS,
            mma_params.tGrKt[None, None, None, smem_pipe_read.index],
            zero_init=False, wg_wait=0,
        )

        # Release both pipelines
        pipeline_k.consumer_release(smem_pipe_read)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    # =========================================================================
    # forward_sub_epilogue: Write dV_i to output buffer and signal completion
    # =========================================================================
    @cute.jit
    def forward_sub_epilogue(
        self,
        acc_O: cute.Tensor,
        mO: cute.Tensor,
        mVorig: cute.Tensor,
        mFlags: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        mTimestamps: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        # Convert acc_O (fp32) to output dtype
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))

        # Store to smem
        smem_copy_atom_O = copy_utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_O = cute.make_tiled_copy_C(
            smem_copy_atom_O, tiled_mma
        ).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        # Fence + barrier for TMA write
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
        )

        # TMA write dV_i to gmem
        gO = cute.local_tile(
            mO[None, None, head_idx, batch_idx],
            (self.tile_m, self.tile_hdimv), (m_block, 0),
        )
        tOsO, tOgO = cpasync.tma_partition(
            tma_atom_O, 0, cute.make_layout(1),
            cute.group_modes(sO, 0, 2), cute.group_modes(gO, 0, 2),
        )

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            cute.copy(tma_atom_O, tOsO, tOgO)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

            # --- Instrument: after TMA write ---
            if const_expr(self.instrument):
                with cute.arch.elect_one():
                    gTs4 = cute.local_tile(mTimestamps, (1, 1), (m_block, 4))
                    store_ts(gTs4.iterator, read_globaltimer())

            # Signal flag: atomic_exch with release semantics
            if const_expr(not self.no_sync):
                cute.arch.fence_acq_rel_gpu()
                head_batch_idx = (
                    head_idx * cute.size(mVorig.shape[3]) + batch_idx
                )
                gFlag = cute.local_tile(
                    mFlags, (1, 1), (head_batch_idx, m_block)
                )
                with cute.arch.elect_one():
                    cute.arch.atomic_exch(
                        gFlag.iterator,
                        Int32(1),
                        sem="release", scope="gpu",
                    )

            # --- Instrument: after signal ---
            if const_expr(self.instrument):
                with cute.arch.elect_one():
                    gTs5 = cute.local_tile(mTimestamps, (1, 1), (m_block, 5))
                    store_ts(gTs5.iterator, read_globaltimer())

    # =========================================================================
    # dk_epilogue: Write acc_dK to HBM via R2S → TMA S2G
    # =========================================================================
    @cute.jit
    def dk_epilogue(
        self,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_dK: cute.CopyAtom,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        # Convert fp32 → bf16/fp16
        rK = cute.make_fragment_like(acc_dK, self.dtype)
        rK.store(acc_dK.load().to(self.dtype))

        # R2S: store to sO (shared memory buffer)
        smem_copy_atom = copy_utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy = cute.make_tiled_copy_C(
            smem_copy_atom, tiled_mma
        ).get_slice(tidx)
        taccOrK = smem_thr_copy.retile(rK)
        taccOsK = smem_thr_copy.partition_D(sO)
        cute.copy(smem_copy_atom, taccOrK, taccOsK)

        # Fence + barrier for smem visibility
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
        )

        # TMA S2G: sO → mdK
        gK = cute.local_tile(
            mdK[None, None, head_idx, batch_idx],
            (self.tile_m, self.tile_hdim), (m_block, 0),
        )
        tKsO, tKgK = cpasync.tma_partition(
            tma_atom_dK, 0, cute.make_layout(1),
            cute.group_modes(sO, 0, 2), cute.group_modes(gK, 0, 2),
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            cute.copy(tma_atom_dK, tKsO, tKgK)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
