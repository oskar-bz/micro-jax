"""`grad` — reverse-mode automatic differentiation (§5.1 of the spec).

The implementation is the textbook tape-based reverse-mode algorithm. The
only non-obvious part is nesting: making `grad(grad(f))` work requires that
the *reverse pass* itself is traceable. We accomplish that by

1. seeding the output cotangent at the *currently active* trace level, and
2. writing every VJP rule in terms of `Array` operations (not raw numpy).

Together that means each `g * cos(x)` inside a VJP is just another `_emit`
call, so any outer tape sees it and can differentiate through it in turn.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .array import Array, ensure_array
from .ops import ones_like
from .tape import current_level, pop_tape, push_tape


def grad(f: Callable[..., Array], argnum: int = 0) -> Callable[..., Array]:
    """Return a function that computes ∂f/∂args[argnum].

    Requirements on `f`:
    * Pure (no side effects, no in-place writes — see §7.1).
    * Returns a scalar `Array` (`ndim == 0` or `size == 1`).

    The returned function has the same call signature as `f` and yields a
    gradient with the same shape as `args[argnum]`.
    """

    def wrapped(*args: Any) -> Array:
        arrays = [ensure_array(a) for a in args]

        # ---- forward pass: record every op onto a fresh tape ----------------
        tape = push_tape()
        # Promote every input to the tape's level so operations involving them
        # are visible to *this* tape (and any outer tapes, since their level
        # is strictly lower).
        traced_args = []
        for a in arrays:
            t = Array.__new__(Array)
            t.data = a.data
            t.id = a.id  # keep id stable so the user can look up grads by it
            t.level = tape.level
            t.batch_dim = a.batch_dim
            traced_args.append(t)
        out = f(*traced_args)
        pop_tape()

        if not isinstance(out, Array):
            raise TypeError(
                "grad: the function must return a micro_jax Array, "
                f"got {type(out).__name__}"
            )
        # `out.size` is the *unbatched* size when running inside a vmap, so
        # this check passes for "scalar per batch element" outputs naturally.
        if out.size != 1:
            raise ValueError(
                "grad: the function must return a (per-sample) scalar; "
                f"got shape {out.shape}"
            )

        # ---- reverse pass: propagate cotangents back through the tape -------
        # Seed dL/dL = 1, at the level of any still-active outer trace so the
        # operations we're about to emit are visible to it.
        seed = ones_like(out)
        seed.level = current_level()
        # Preserve the batch tag — under vmap, the seed itself must carry a
        # batch_dim so each VJP step stays batched too.
        seed.batch_dim = out.batch_dim
        cotangents: dict[int, Array] = {out.id: seed}

        for entry in reversed(tape):
            g = cotangents.get(entry.output_id)
            if g is None:
                # This entry's output is not on the path to the loss — its
                # cotangent is implicitly zero, skip it.
                continue

            # Wrap saved primal values as fresh Arrays — but reuse the
            # *original* input id. The reverse pass itself emits operations
            # via `_emit`, and under jit those ops are recorded by id; if
            # we minted fresh ids here, the abstract recorder would treat
            # every primal as a brand-new constant and lose the connection
            # back to the trace inputs.
            primals = []
            for inp_id, inp_val, inp_bd in zip(
                entry.input_ids, entry.input_vals, entry.input_batch_dims
            ):
                p = Array.__new__(Array)
                p.data = np.asarray(inp_val, dtype=np.float32)
                p.id = inp_id
                # Tag primals with the *currently active* level (= the level
                # of any still-active outer tape). This makes VJP-internal
                # operations that involve only primals — e.g. `power(x, n-1)`
                # in pow's VJP — visible to the outer tape, which is what
                # makes higher-order differentiation work.
                p.level = current_level()
                p.batch_dim = inp_bd
                primals.append(p)
            input_grads = entry.prim.vjp(g, *primals, **entry.params)

            for input_id, grad_val in zip(entry.input_ids, input_grads):
                if input_id in cotangents:
                    cotangents[input_id] = cotangents[input_id] + grad_val
                else:
                    cotangents[input_id] = grad_val

        target_id = traced_args[argnum].id
        if target_id not in cotangents:
            # Output didn't actually depend on this argument — the gradient
            # is structurally zero.
            return Array(0.0 * traced_args[argnum].data)
        return cotangents[target_id]

    return wrapped


def value_and_grad(
    f: Callable[..., Array], argnum: int = 0
) -> Callable[..., tuple[Array, Array]]:
    """Convenience wrapper returning `(f(*args), grad(f)(*args))`."""

    def wrapped(*args: Any) -> tuple[Array, Array]:
        arrays = [ensure_array(a) for a in args]
        value = f(*arrays)
        gradient = grad(f, argnum)(*arrays)
        return value, gradient

    return wrapped
