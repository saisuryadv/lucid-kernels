# Forward Substitution kernel — ported from LUCID to upstream FA3 patterns.
#
# Implements tiled forward substitution:
#   V'_i = inv(tril(exp(K_i@K_i^T))) @ (V_i - sum_{j<i} exp(K_i @ K_j^T) @ V'_j)
#
# Forked from flash_fwd_sm90.py (upstream FA3).  Key changes vs FA3:
#   1. Replace softmax with element-wise -exp in sub_one_n_block
#   2. Loop over j=0..i-1 in forward order (farthest first)
#   3. Epilogue: TMA write V'_i + atomic flag signal
#   4. Flag synchronization for inter-block data dependencies
#   5. Diagonal block solve via WGMMA (diag_inv @ RHS)
#   6. TMA descriptors: Q-slot→K, K-slot→K, V-slot→V', O-slot→V', D-slot→diag_inv

import math
from types import SimpleNamespace
from typing import Type, Callable, Optional
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


class ForwardSubSm90:
    """Forward substitution kernel using upstream FA3's SM90 WGMMA/TMA infrastructure.

    Computes V'_i = inv(M_ii) @ (V_i - sum_{j<i} exp(K_i @ K_j^T) @ V'_j)
    where M_ii = tril(exp(K_i @ K_i^T)).
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
        window_size: Optional[int] = None,
        no_sync: bool = False,
        instrument: bool = False,
        use_diag_solve: bool = False,
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
        self.intra_wg_overlap = False  # Phase 1: no overlap
        self.window_size = window_size
        self.no_sync = no_sync
        self.instrument = instrument
        self.use_diag_solve = use_diag_solve
        self.buffer_align_bytes = 1024

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        sP_layout_atom = None  # mma_pv_is_rs = True
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        # QK GEMM: K_i @ K_j^T → (m_block, n_block)
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        # PV GEMM: P @ V' → (m_block, head_dim_v). A from registers (mma_pv_is_rs=True).
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM,
        )
        # Diagonal solve: diag_inv(smem, K-major) @ RHS^T(smem, MN-major) → acc_O
        tiled_mma_diag = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_hdimv),
        )
        return tiled_mma_qk, tiled_mma_pv, tiled_mma_diag

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

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr: mbar_ptr_QO_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_diag: mbar_diag_struct
            sV_recv: sV_recv_struct
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
        mVorig: cute.Tensor,   # (batch, seqlen, heads_kv, head_dim_v) — original V
        mTimestamps: cute.Tensor,  # (T, 8) int64 — debug timestamps
        mDiagInv: cute.Tensor, # (batch, seqlen, heads_kv, block_size) — diag_inv
        stream: cuda.CUstream,
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

        tiled_mma_qk, tiled_mma_pv, tiled_mma_diag = self._get_tiled_mma()
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

        # Create smem layouts using upstream pattern
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
            for mX, shape, stage in [
                (mK, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mVprime, (self.tile_n, self.tile_hdimv), self.num_stages),
                (mVprime, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        # sV_recv layout: same as sO (single buffer for diag_inv)
        self.sV_recv_layout = self.sO_layout

        SharedStorage = self._get_shared_storage_cls()

        # TMA descriptors
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mK, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mVprime, self.sV_layout),
            ]
        }

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
        # V-slot TMA → loads V'_j from V' buffer
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV, mVprime,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv), 1,
        )
        # O-slot TMA → writes V'_i to V' buffer
        tma_atom_O, tma_tensor_Vprime = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_O, mVprime, self.sO_layout, (self.tile_m, self.tile_hdimv),
        )
        # D-slot TMA → loads diag_inv_i into sV_recv
        tma_atom_D, tma_tensor_D = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q, mDiagInv, self.sV_recv_layout,
            (self.tile_m, self.tile_hdimv),
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
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            tma_atom_D,
            Float32(LOG2_E),
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sV_recv_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_diag,
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
    # kernel: CUDA kernel entry — producer/consumer warp specialization
    # =========================================================================
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,       # K data (Q-slot TMA tensor)
        mK: cute.Tensor,       # K data (K-slot TMA tensor)
        mV: cute.Tensor,       # V' data (V-slot TMA tensor)
        mO: cute.Tensor,       # V' data (O-slot TMA tensor, for output)
        mVorig: cute.Tensor,   # Original V
        mFlags: cute.Tensor,   # Flags (heads_kv * batch, T)
        mTimestamps: cute.Tensor,
        mDiagInv: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        tma_atom_D: cute.CopyAtom,
        log2e: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sV_recv_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_diag: cute.TiledMma,
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

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier init for Q-slot (K_i load)
        mbar_ptr_Q = storage.mbar_ptr.data_ptr()
        if warp_idx == 1:
            cute.arch.mbarrier_init(mbar_ptr_Q, 1)

        # Diag solve mbarrier init
        mbar_diag_ptr = storage.mbar_diag.data_ptr()
        if const_expr(self.use_diag_solve):
            if warp_idx == 1:
                cute.arch.mbarrier_init(mbar_diag_ptr, 1)

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
            defer_sync=False,
        )

        # Get shared memory buffers
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = layout_utils.transpose_view(sV)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

        # sV_recv buffer for diag_inv
        sV_recv = storage.sV_recv.get_tensor(sV_recv_layout.outer, swizzle=sV_recv_layout.inner, dtype=self.dtype)

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
            )
        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk, tiled_mma_pv, tiled_mma_diag,
                mO, mVorig, mFlags,
                sQ, sK, sVt, sO,
                pipeline_k, pipeline_v, mbar_ptr_Q,
                tma_atom_O, mTimestamps,
                tidx, log2e,
                TileSchedulerCls,
                sV_recv, mbar_diag_ptr,
            )

    # =========================================================================
    # load: Producer — loads K_i (Q-slot) and streams K_j, V'_j via TMA
    # =========================================================================
    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,   # K data via Q-slot
        mK: cute.Tensor,   # K data via K-slot
        mV: cute.Tensor,   # V' data via V-slot
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
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            kv_producer_state = pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx

                # TMA partition setup for this tile
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

                # Load K_i into Q-slot smem
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr_Q, self.tma_copy_bytes["Q"])
                cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=mbar_ptr_Q)

                # Preload diag_inv_i into sV_recv — overlaps with inner loop
                if const_expr(self.use_diag_solve):
                    mDiagInv_cur = mDiagInv[None, None, head_idx, batch_idx]
                    gDiag = cute.local_tile(mDiagInv_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
                    tDsD, tDgD = cpasync.tma_partition(
                        tma_atom_D, 0, cute.make_layout(1),
                        cute.group_modes(sV_recv, 0, 2), cute.group_modes(gDiag, 0, 2),
                    )
                    diag_copy_bytes = self.tile_m * self.tile_hdimv * (self.dtype.width // 8)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_diag_ptr, diag_copy_bytes)
                    cute.copy(tma_atom_D, tDgD, tDsD, tma_bar_ptr=mbar_diag_ptr)

                # Instrument: producer before inner loop
                if const_expr(self.instrument):
                    with cute.arch.elect_one():
                        gTs0 = cute.local_tile(mTimestamps, (1, 1), (m_block, 0))
                        store_ts(gTs0.iterator, read_globaltimer())

                # Forward order: farthest block first
                if const_expr(self.window_size is not None):
                    n_block_count = cutlass.min(m_block, self.window_size)
                else:
                    n_block_count = m_block
                n_block_start = m_block - n_block_count

                for j_iter in cutlass.range(n_block_count, unroll=2):
                    n_block = n_block_start + j_iter

                    # Load K_j
                    pipeline_k.producer_acquire(kv_producer_state)
                    cute.copy(
                        tma_atom_K,
                        tKgK[None, n_block],
                        tKsK[None, kv_producer_state.index],
                        tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state),
                    )

                    # Wait for flag from CTA j before loading V'_j
                    if const_expr(not self.no_sync):
                        head_batch_idx = head_idx * cute.size(mK.shape[3]) + batch_idx
                        gFlag = cute.local_tile(mFlags, (1, 1), (head_batch_idx, n_block))
                        flag_ptr = gFlag.iterator
                        with cute.arch.elect_one():
                            flag_val = cute.arch.atomic_add(flag_ptr, Int32(0), sem="acquire", scope="gpu")
                            while flag_val == 0:
                                flag_val = cute.arch.atomic_add(flag_ptr, Int32(0), sem="acquire", scope="gpu")

                    # Load V'_j
                    pipeline_v.producer_acquire(kv_producer_state)
                    cute.copy(
                        tma_atom_V,
                        tVgV[None, n_block],
                        tVsV[None, kv_producer_state.index],
                        tma_bar_ptr=pipeline_v.producer_get_barrier(kv_producer_state),
                    )

                    kv_producer_state.advance()

                # Instrument: producer after inner loop
                if const_expr(self.instrument):
                    with cute.arch.elect_one():
                        gTs1 = cute.local_tile(mTimestamps, (1, 1), (m_block, 1))
                        store_ts(gTs1.iterator, read_globaltimer())

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    # =========================================================================
    # mma: Consumer — accumulates corrections and writes V'_i
    # =========================================================================
    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_diag: cute.TiledMma,
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
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )

        # QK GEMM fragments
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        # PV GEMM fragments (sP=None since mma_pv_is_rs=True)
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), None, sVt
        )

        # Diagonal solve fragments
        if const_expr(self.use_diag_solve):
            wg_mma_diag = tiled_mma_diag.get_slice(warp_group_thread_layout(warp_group_idx))
            tDrDiag = tiled_mma_diag.make_fragment_A(wg_mma_diag.partition_A(sV_recv))
            sOt = layout_utils.transpose_view(sO)
            tDrRHSt = tiled_mma_diag.make_fragment_B(wg_mma_diag.partition_B(sOt))

        # Number of rows for element-wise exp loop
        acc_S_shape = tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        num_rows = acc_S_shape[0][0] * acc_S_shape[1]

        mma_params = SimpleNamespace(tSrQ=tSrQ, tSrK=tSrK, tOrP=tOrP, tOrVt=tOrVt, acc_O=acc_O)

        q_consumer_phase = Int32(0)
        diag_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx

            # Wait for K_i (Q-slot) to be loaded
            cute.arch.mbarrier_wait(mbar_ptr_Q, phase=q_consumer_phase)
            q_consumer_phase ^= 1

            # Instrument: consumer after K_i loaded
            if const_expr(self.instrument):
                with cute.arch.elect_one():
                    gTs2 = cute.local_tile(mTimestamps, (1, 1), (m_block, 2))
                    store_ts(gTs2.iterator, read_globaltimer())

            # Initialize acc_O = V_i (prefetch from HBM before inner loop).
            # P = -exp(S), WGMMA accumulates acc_O += P @ V'_j = V_i - Σ exp(S) @ V'_j
            mVorig_cur = mVorig[None, None, head_idx, batch_idx]
            gVi = cute.local_tile(mVorig_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            thr_mma_pv = tiled_mma_pv.get_slice(tidx)
            taccOgVi = thr_mma_pv.partition_C(gVi)
            acc_O.store(taccOgVi.load().to(Float32))

            # Compute window bounds
            if const_expr(self.window_size is not None):
                n_block_count = cutlass.min(m_block, self.window_size)
            else:
                n_block_count = m_block
            n_block_start = m_block - n_block_count

            # Inner loop: accumulate corrections in forward order
            for j_iter in cutlass.range(n_block_count, unroll=1):
                n_block = n_block_start + j_iter
                kv_consumer_state = self.sub_one_n_block(
                    n_block, kv_consumer_state,
                    tiled_mma_qk, tiled_mma_pv,
                    pipeline_k, pipeline_v,
                    mma_params, log2e, num_rows,
                )

            # Instrument: consumer after MMA loop
            if const_expr(self.instrument):
                with cute.arch.elect_one():
                    gTs3 = cute.local_tile(mTimestamps, (1, 1), (m_block, 3))
                    store_ts(gTs3.iterator, read_globaltimer())

            # Diagonal solve: V'_i = diag_inv_i @ RHS_i
            if const_expr(self.use_diag_solve):
                # 1. Store RHS (acc_O) → sO as bf16/fp16
                rO_diag = cute.make_fragment_like(acc_O, self.dtype)
                rO_diag.store(acc_O.load().to(self.dtype))
                smem_copy_atom_diag = copy_utils.get_smem_store_atom(self.arch, self.dtype)
                smem_thr_copy_diag = cute.make_tiled_copy_C(smem_copy_atom_diag, tiled_mma_pv).get_slice(tidx)
                taccOrO_diag = smem_thr_copy_diag.retile(rO_diag)
                taccOsO_diag = smem_thr_copy_diag.partition_D(sO)
                cute.copy(smem_copy_atom_diag, taccOrO_diag, taccOsO_diag)

                # 2. Fence + barrier sync
                cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                cute.arch.barrier(barrier_id=DIAG_SYNC_BARRIER_ID,
                                  number_of_threads=self.num_epilogue_threads)

                # 3. Wait for diag_inv TMA load into sV_recv
                cute.arch.mbarrier_wait(mbar_diag_ptr, phase=diag_consumer_phase)
                diag_consumer_phase ^= 1

                # 4. WGMMA: acc_O = diag_inv @ RHS^T
                sm90_utils.gemm(tiled_mma_diag, acc_O, tDrDiag, tDrRHSt,
                                zero_init=True, wg_wait=0)

            # Instrument: after diag_solve, before epilogue
            if const_expr(self.instrument):
                with cute.arch.elect_one():
                    gTs6 = cute.local_tile(mTimestamps, (1, 1), (m_block, 6))
                    store_ts(gTs6.iterator, read_globaltimer())

            # Epilogue: write V'_i and signal flag
            self.forward_sub_epilogue(
                acc_O, mO, mVorig, mFlags,
                sO, tma_atom_O, mTimestamps,
                tiled_mma_pv, tidx,
                m_block, head_idx, batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # =========================================================================
    # sub_one_n_block: Process one off-diagonal block
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
    ):
        # Allocate S accumulator
        acc_S = cute.make_fragment(
            tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )

        # Wait for K_j, compute S = K_i @ K_j^T
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        sm90_utils.gemm(
            tiled_mma_qk, acc_S, mma_params.tSrQ,
            mma_params.tSrK[None, None, None, smem_pipe_read.index],
            zero_init=True, wg_wait=-1
        )
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        # Element-wise -exp: P = -exp(S) = -exp2(S * log2(e))
        acc_S_mn = layout_utils.make_acc_tensor_mn_view(acc_S)
        for r in cutlass.range(num_rows, unroll_full=True):
            row = acc_S_mn[r, None].load()
            acc_S_mn[r, None].store(-cute.math.exp2(row * log2e, fastmath=True))

        # Convert to fp16/bf16
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        utils.cvt_f16(tOrP_acc, mma_params.tOrP)

        # Wait for V'_j, compute acc_O += P @ V'_j
        pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
        sm90_utils.gemm(
            tiled_mma_pv, mma_params.acc_O, mma_params.tOrP,
            mma_params.tOrVt[None, None, None, smem_pipe_read.index],
            zero_init=False, wg_wait=0
        )
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    # =========================================================================
    # forward_sub_epilogue: Write V'_i to V' buffer and signal flag
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
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        # Fence + barrier for TMA write
        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
        cute.arch.barrier_arrive(barrier_id=int(NamedBarrierFwd.Epilogue),
                                 number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE)

        # TMA write V'_i to gmem
        gO = cute.local_tile(mO[None, None, head_idx, batch_idx],
                             (self.tile_m, self.tile_hdimv), (m_block, 0))
        tOsO, tOgO = cpasync.tma_partition(
            tma_atom_O, 0, cute.make_layout(1),
            cute.group_modes(sO, 0, 2), cute.group_modes(gO, 0, 2),
        )

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue),
                              number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE)
            cute.copy(tma_atom_O, tOsO, tOgO)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

            # Instrument: after TMA write
            if const_expr(self.instrument):
                with cute.arch.elect_one():
                    gTs4 = cute.local_tile(mTimestamps, (1, 1), (m_block, 4))
                    store_ts(gTs4.iterator, read_globaltimer())

            # Signal flag: atomic_exch with release semantics
            if const_expr(not self.no_sync):
                cute.arch.fence_acq_rel_gpu()
                head_batch_idx = head_idx * cute.size(mVorig.shape[3]) + batch_idx
                gFlag = cute.local_tile(mFlags, (1, 1), (head_batch_idx, m_block))
                with cute.arch.elect_one():
                    cute.arch.atomic_exch(
                        gFlag.iterator,
                        Int32(1),
                        sem="release", scope="gpu",
                    )

            # Instrument: after signal
            if const_expr(self.instrument):
                with cute.arch.elect_one():
                    gTs5 = cute.local_tile(mTimestamps, (1, 1), (m_block, 5))
                    store_ts(gTs5.iterator, read_globaltimer())
