"""Primitives and the central dispatcher (`_emit`).

A `Primitive` is the atomic computational unit of the system. Each one bundles:

* `fwd`           — concrete forward evaluation on numpy arrays.
* `vjp`           — vector-Jacobian-product rule (reverse-mode AD).
* `abstract_eval` — output shape given input shapes (used by jit).
* `batch_rule`    — how the primitive lifts under `vmap`. Optional; a sensible
                    elementwise default is used when not provided.

The whole system funnels every `Array → Array` operation through `_emit`. That
single function decides:

1. Is an *abstract trace* currently capturing operations? (jit)
2. Do any inputs carry a `batch_dim`? (vmap)
3. Which active tapes should record this op? (grad — possibly nested)

Keeping the dispatch logic centralized is what lets the various transforms
compose without knowing about one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .array import Array, _new_id, ensure_array
from .tape import active_tapes


# A registry of every known primitive, keyed by name. Useful for debugging and
# for the abstract-tape executor in `jit.py`.
REGISTRY: dict[str, "Primitive"] = {}


# A stack of active "abstract recorders". Each abstract recorder intercepts
# `_emit` calls so they build an `AbstractTape` instead of running concretely.
# `jit.py` pushes a recorder for the duration of its tracing pass.
_ABSTRACT_STACK: list["AbstractRecorder"] = []


class AbstractRecorder:
    """Hook interface for transforms that want to intercept _emit calls.

    Concrete implementation lives in `jit.py`. Defined here only as a typing
    anchor so `_emit` can dispatch to it without importing `jit`.
    """

    def record(
        self, prim: "Primitive", inputs: list[Array], params: dict[str, Any]
    ) -> Array:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class Primitive:
    name: str
    fwd: Callable[..., np.ndarray]
    vjp: Callable[..., tuple[Array, ...]]
    abstract_eval: Callable[..., tuple[int, ...]] | None = None
    batch_rule: Callable[..., tuple[np.ndarray, int | None]] | None = None

    def __post_init__(self) -> None:
        REGISTRY[self.name] = self

    def bind(self, *args: Any, **params: Any) -> Array:
        """Invoke this primitive on the given inputs and parameters."""
        return _emit(self, list(args), params)

    def __repr__(self) -> str:
        return f"Primitive({self.name!r})"


# ---------------------------------------------------------------------------
# Default rules
# ---------------------------------------------------------------------------


def _default_abstract_eval(prim: Primitive):
    """Shape inference by running `fwd` on zero-filled placeholders.

    Works for any primitive whose output shape depends only on input shapes —
    which covers every built-in. Custom primitives that depend on values (rare)
    must override `abstract_eval` explicitly.
    """

    def rule(*input_shapes: tuple[int, ...], **params: Any) -> tuple[int, ...]:
        dummies = [np.zeros(s, dtype=np.float32) for s in input_shapes]
        return tuple(prim.fwd(*dummies, **params).shape)

    return rule


def _default_batch_rule(prim: Primitive):
    """Generic batch rule that works for any primitive whose underlying numpy
    operation naturally broadcasts the batch axis.

    Strategy: normalize each batched input so its batch axis sits at position
    0, then call `fwd` directly. Numpy's elementwise rules (and `np.matmul`'s
    broadcasting over leading dims) handle the rest. The output's batch axis
    is also at position 0.
    """

    def rule(
        input_data: list[np.ndarray],
        batch_dims: list[int | None],
        params: dict[str, Any],
    ) -> tuple[np.ndarray, int | None]:
        moved = []
        for x, bd in zip(input_data, batch_dims):
            if bd is None or bd == 0:
                moved.append(x)
            else:
                moved.append(np.moveaxis(x, bd, 0))
        return prim.fwd(*moved, **params), 0

    return rule


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _emit(
    prim: Primitive,
    inputs: list[Any],
    params: dict[str, Any] | None = None,
) -> Array:
    """Apply a primitive. The single chokepoint in the system.

    Order of decisions:

    1. **Abstract tracing (jit)** — if active, the recorder owns shape
       inference and we skip the concrete forward pass entirely.
    2. **vmap batching** — if any input has a non-None batch_dim, dispatch
       to the primitive's batch rule (with a sensible elementwise default).
    3. **Plain eager evaluation** — the default.

    Recording on grad tapes happens *after* the output is built, regardless
    of whether the value came from a concrete fwd or from abstract tracing.
    That uniformity is what makes `jit(grad(f))` work: the grad tape and
    the jit abstract tape both see every operation.
    """
    params = params or {}
    inputs = [ensure_array(x) for x in inputs]

    if _ABSTRACT_STACK:
        # jit owns evaluation. It returns an Array whose `data` is a
        # placeholder of the correct shape — we never look at the placeholder
        # values, but we *do* still record onto active grad tapes below.
        out = _ABSTRACT_STACK[-1].record(prim, inputs, params)
    else:
        batch_dims = [x.batch_dim for x in inputs]
        if any(b is not None for b in batch_dims):
            rule = prim.batch_rule or _default_batch_rule(prim)
            out_data, out_batch_dim = rule(
                [x.data for x in inputs], batch_dims, params
            )
        else:
            out_data = prim.fwd(*(x.data for x in inputs), **params)
            out_batch_dim = None

        out = Array.__new__(Array)
        out.data = np.asarray(out_data, dtype=np.float32)
        out.id = _new_id()
        out.batch_dim = out_batch_dim

    # Output's trace level is the max of its inputs' levels (§3.3).
    out_level = max((x.level for x in inputs), default=0)
    out.level = out_level

    # Record on every active tape whose level is at or below the output's
    # level. The level check is what makes nested grads (`grad(grad(f))`)
    # behave correctly — the outer tape sees inner-tape operations because
    # those ops live at a >= level, but the outer reverse pass operates at
    # the outer level so its own ops won't pollute the inner tape (which is
    # already popped by then anyway).
    for tape in active_tapes():
        if tape.level <= out_level:
            tape.record(prim, inputs, out, params)

    return out


# ---------------------------------------------------------------------------
# Abstract-recorder helpers (used by jit.py)
# ---------------------------------------------------------------------------


def push_abstract_recorder(recorder: AbstractRecorder) -> None:
    _ABSTRACT_STACK.append(recorder)


def pop_abstract_recorder() -> AbstractRecorder:
    return _ABSTRACT_STACK.pop()


def in_abstract_trace() -> bool:
    return bool(_ABSTRACT_STACK)
