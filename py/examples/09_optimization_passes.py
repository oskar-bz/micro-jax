"""See `jit`'s optimization passes (DCE -> const-fold -> CSE) on a small
program.

We construct a function with deliberate inefficiencies:

* `sin(x)` is computed twice            (CSE will merge them)
* `a + b` doesn't depend on `x`         (const-fold will collapse it)
* `exp(x) * cos(x)` is unused           (DCE will drop it)

Then we run the same trace through each pass in isolation and print the
shrinking program. No numpy magic here -- `_trace`, `_dce`, `_const_fold`,
`_cse` are the real internals.
"""

from __future__ import annotations

import copy

import numpy as np

import micro_jax as mjx
from micro_jax import Array
from micro_jax.jit import (
    AbstractEntry,
    CompiledProgram,
    _const_fold,
    _cse,
    _dce,
    _dedup_constants,
    _trace,
)


def print_program(prog: CompiledProgram, title: str) -> None:
    rename: dict[int, str] = {}

    def short(i: int) -> str:
        if i not in rename:
            rename[i] = f"v{len(rename)}"
        return rename[i]

    # Pre-name inputs and constants so they read nicely in the output.
    for i in prog.input_ids:
        short(i)
    for i in prog.constants:
        short(i)

    print(f"\n=== {title}  ({len(prog.entries)} ops) ===")
    print(f"  inputs:    {[short(i) for i in prog.input_ids]}")
    if prog.constants:
        bits = []
        for cid, val in prog.constants.items():
            v = val.item() if val.size == 1 else val.tolist()
            bits.append(f"{short(cid)}={v}")
        print(f"  constants: [{', '.join(bits)}]")
    for e in prog.entries:
        out = short(e.output_id)
        ins = ", ".join(short(i) for i in e.input_ids)
        ps = ", ".join(f"{k}={v}" for k, v in e.params.items())
        suffix = f" [{ps}]" if ps else ""
        print(f"  {out:>4} = {e.prim.name:<12}({ins}){suffix}")
    print(f"  output:    {short(prog.output_id)}")


def copy_program(prog: CompiledProgram) -> CompiledProgram:
    return CompiledProgram(
        input_ids=prog.input_ids,
        entries=[
            AbstractEntry(
                prim=e.prim,
                input_ids=e.input_ids,
                output_id=e.output_id,
                output_shape=e.output_shape,
                params=dict(e.params),
            )
            for e in prog.entries
        ],
        output_id=prog.output_id,
        constants={k: v.copy() for k, v in prog.constants.items()},
    )


def main() -> None:
    # A handful of captured constants the compiler can statically evaluate.
    a = Array(2.0)
    b = Array(3.0)
    c = Array(0.5)
    weights = Array(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def f(x: Array) -> Array:
        # --- a tower of pure-constant arithmetic (all foldable) -----------
        k1 = a + b               # 5
        k2 = a * b               # 6
        k3 = k1 * k2 + c         # 30.5
        k4 = mjx.sin(k1) * mjx.cos(k2)
        k5 = mjx.exp(c) + k1     # e^0.5 + 5

        # --- repeated subexpressions of `x` (CSE will collapse to one) ----
        s1 = mjx.sin(x)
        s2 = mjx.sin(x)
        s3 = mjx.sin(x)
        e1 = mjx.exp(x * Array(2.0))
        e2 = mjx.exp(x * Array(2.0))

        # --- compound subexpressions that become identical *after*
        #     lower-level CSE has merged their pieces -----------------------
        p1 = s1 + e1
        p2 = s2 + e2
        p3 = s3 + e1

        # --- four dead branches: none reaches the output ------------------
        dead_a = mjx.log(x + Array(10.0))
        dead_b = mjx.cos(x) * mjx.sin(x)
        dead_c = (dead_a + dead_b) * weights
        dead_d = mjx.matmul(
            mjx.reshape(dead_c, (1, 3)),
            mjx.reshape(weights, (3, 1)),
        )
        _ = dead_d  # silence linters; the optimizer should still drop it

        # --- mix in some folded-constant * traced-input ops ---------------
        scaled = x * k3 + k4
        return mjx.sum(p1 * p2 + p3 + scaled + weights * k5)

    # Trace once. `_trace` runs `f` with abstract inputs and returns the raw
    # program. We then run each pass on a copy so we can show before/after.
    trace_input = Array(np.zeros((3,), dtype=np.float32))
    raw = _trace(f, (trace_input,))
    print_program(raw, "1. Raw trace")

    p = copy_program(raw)
    _dce(p)
    print_program(p, "2. After DCE")

    _const_fold(p)
    print_program(p, "3. After constant folding")

    _dedup_constants(p)
    print_program(p, "4. After constant deduplication")

    _cse(p)
    print_program(p, "5. After CSE")

    # The end-to-end optimizer also runs a final DCE to mop up any entries
    # that CSE made unreachable. Demonstrate it produces the same program
    # as the public `jit`.
    jf = mjx.jit(f)
    _ = jf(Array(1.5))
    public_program = next(iter(jf._cache.values()))
    print_program(
        public_program,
        "6. Public jit cache (DCE -> fold -> dedup -> CSE -> DCE)",
    )


if __name__ == "__main__":
    main()
