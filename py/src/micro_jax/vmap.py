"""`vmap` — automatic vectorization via batch tracers (§5.3 of the spec).

Idea: tag each input with the axis along which it is batched. Then every
primitive's `batch_rule` says how to compute its result given the tag, *as
if* we'd run a python `for` loop — but in one fused numpy call.

Unlike `grad`, `vmap` doesn't keep a tape. The information it needs flows on
the `Array.batch_dim` field, which `_emit` reads on every call. The default
batch rule in `primitives.py` already handles every elementwise primitive
(numpy auto-broadcasts along leading dims). Shape-changing primitives carry
their own rules (see `ops.py`).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .array import Array, ensure_array


def _normalize_in_axes(
    in_axes: int | None | Sequence[int | None], n_args: int
) -> list[int | None]:
    if isinstance(in_axes, (tuple, list)):
        if len(in_axes) != n_args:
            raise ValueError(
                f"vmap: in_axes has {len(in_axes)} entries, function has {n_args} args"
            )
        return list(in_axes)
    return [in_axes] * n_args


def vmap(
    f: Callable[..., Array],
    in_axes: int | None | Sequence[int | None] = 0,
    out_axis: int = 0,
) -> Callable[..., Array]:
    """Vectorize `f` over a batch dimension.

    Each argument is tagged with the axis along which it carries a batch
    (or `None` to broadcast). Tags ride on the `Array` objects through
    every primitive, and each primitive's batch rule says how to compute
    its result in one shot.

    The output's batch axis lands at `out_axis` (default 0). If the function
    happens to produce a value that didn't depend on any batched input,
    we broadcast it across the batch — matching the obvious "what would a
    python `for` loop have done" semantics.
    """

    def wrapped(*args: Any) -> Array:
        arrs = [ensure_array(a) for a in args]
        axes = _normalize_in_axes(in_axes, len(arrs))

        # Establish the batch size by reading the first non-None axis.
        batch_size: int | None = None
        for arr, ax in zip(arrs, axes):
            if ax is None:
                continue
            n = arr.shape[ax]
            if batch_size is None:
                batch_size = n
            elif batch_size != n:
                raise ValueError(
                    f"vmap: inconsistent batch sizes: {batch_size} vs {n}"
                )
        if batch_size is None:
            raise ValueError("vmap: at least one input must be batched")

        # Tag inputs without copying their data.
        tagged: list[Array] = []
        for arr, ax in zip(arrs, axes):
            t = Array.__new__(Array)
            t.data = arr.data
            t.id = arr.id  # reuse id so grad-inside-vmap can find inputs
            t.level = arr.level
            t.batch_dim = ax
            tagged.append(t)

        out = f(*tagged)
        if not isinstance(out, Array):
            raise TypeError(
                f"vmap: function must return Array, got {type(out).__name__}"
            )

        # If the output never picked up a batch_dim, the function ignored
        # the batched inputs — replicate the scalar/non-batched result so
        # the user sees an `(N, ...)` shape as they'd expect.
        if out.batch_dim is None:
            data = np.broadcast_to(out.data, (batch_size,) + out.shape).copy()
            return Array(data)

        # Move the batch axis to the requested output position.
        if out.batch_dim != out_axis:
            data = np.moveaxis(out.data, out.batch_dim, out_axis)
        else:
            data = out.data
        return Array(data)

    return wrapped
