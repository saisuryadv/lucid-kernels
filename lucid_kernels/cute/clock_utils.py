# Clock utilities for kernel instrumentation.
# Provides globaltimer reads and timestamp store via inline PTX.
# Copied from lucid_flash_attn for forward substitution kernel.

import cutlass
from cutlass import dsl_user_op, Int64, Int32
from cutlass._mlir.dialects import llvm, nvvm
from cutlass._mlir import ir

T = ir.IntegerType.get_signless


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read the GPU-wide globaltimer (64-bit nanosecond counter)."""
    return Int64(nvvm.read_ptx_sreg_globaltimer(T(64), loc=loc, ip=ip))


@dsl_user_op
def store_ts(ptr, value: Int64, *, loc=None, ip=None) -> None:
    """Store a 64-bit value to a global memory pointer via st.global.u64."""
    llvm.inline_asm(
        None,
        [
            ptr.to_llvm_ptr(loc=loc, ip=ip),
            Int64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
