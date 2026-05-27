"""The core value type.

`Array` is what flows through every micro-jax computation. Internally it is just
a numpy array carrying two pieces of trace metadata:

* `level`     — which trace nesting level the value belongs to (§4.3 of the spec).
                Concrete, untraced values are level 0.
* `batch_dim` — for `vmap`: which axis of `data` represents the batch (§5.3).
                None for non-batched values.

Both default to "no tracing" so an `Array` is just an f32 numpy array with an id.

The class deliberately stays small and dumb. All cleverness — recording on tapes,
applying batch rules, broadcasting — lives in `primitives._emit` and the
transforms. This keeps the type easy to reason about.
"""

from __future__ import annotations

from itertools import count
from typing import Any

import numpy as np

# Monotonic global id counter. Each `Array` gets a unique, immutable id at
# allocation time. The reverse pass uses these ids as dictionary keys to look
# up accumulated cotangents (§7.3).
_id_counter = count()


def _new_id() -> int:
    return next(_id_counter)


def _as_f32(data: Any) -> np.ndarray:
    """Coerce arbitrary numeric input to a contiguous f32 numpy array."""
    if isinstance(data, Array):
        return data.data
    arr = np.asarray(data, dtype=np.float32)
    # numpy returns 0-d arrays for python scalars — that matches the spec's
    # "scalar array" (§1.2 ndim=0) so we leave them as-is.
    return arr


class Array:
    """An n-dimensional f32 array with trace metadata."""

    __slots__ = ("data", "id", "level", "batch_dim")

    def __init__(
        self,
        data: Any,
        level: int = 0,
        batch_dim: int | None = None,
    ) -> None:
        self.data = _as_f32(data)
        self.id = _new_id()
        self.level = level
        self.batch_dim = batch_dim

    # --- views over the underlying numpy buffer ------------------------------
    # Under `vmap`, the buffer has an extra batch axis at `batch_dim`. The
    # tracer-level view of the Array hides that axis — `shape`, `ndim`, and
    # `size` report the per-sample dimensions, which is what user code and
    # VJP rules need to reason about. Primitive batch rules go through
    # `data` directly to see the full buffer.

    @property
    def shape(self) -> tuple[int, ...]:
        s = self.data.shape
        if self.batch_dim is None:
            return s
        return s[: self.batch_dim] + s[self.batch_dim + 1 :]

    @property
    def ndim(self) -> int:
        return self.data.ndim - (0 if self.batch_dim is None else 1)

    @property
    def size(self) -> int:
        if self.batch_dim is None:
            return self.data.size
        return self.data.size // self.data.shape[self.batch_dim]

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __repr__(self) -> str:
        tag = ""
        if self.level:
            tag += f" level={self.level}"
        if self.batch_dim is not None:
            tag += f" batch_dim={self.batch_dim}"
        return f"Array({self.data!r}{tag})"

    def __float__(self) -> float:
        # Convenience for tests / scalar checks. Refuses to lie about
        # non-scalar arrays.
        if self.size != 1:
            raise TypeError("only size-1 Arrays can be converted to a python float")
        return float(self.data.reshape(()))

    # --- operator overloads --------------------------------------------------
    # All of these go through the registered primitives, so trace recording
    # and vmap-style batching happen automatically.

    def __add__(self, other: Any) -> Array:
        from .ops import add

        return add(self, other)

    def __radd__(self, other: Any) -> Array:
        from .ops import add

        return add(other, self)

    def __sub__(self, other: Any) -> Array:
        from .ops import sub

        return sub(self, other)

    def __rsub__(self, other: Any) -> Array:
        from .ops import sub

        return sub(other, self)

    def __mul__(self, other: Any) -> Array:
        from .ops import mul

        return mul(self, other)

    def __rmul__(self, other: Any) -> Array:
        from .ops import mul

        return mul(other, self)

    def __truediv__(self, other: Any) -> Array:
        from .ops import div

        return div(self, other)

    def __rtruediv__(self, other: Any) -> Array:
        from .ops import div

        return div(other, self)

    def __neg__(self) -> Array:
        from .ops import neg

        return neg(self)

    def __pow__(self, exponent: float | int) -> Array:
        from .ops import power

        return power(self, exponent)

    def __matmul__(self, other: Any) -> Array:
        from .ops import matmul

        return matmul(self, other)

    def __rmatmul__(self, other: Any) -> Array:
        from .ops import matmul

        return matmul(other, self)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> Array:
        from .ops import asum

        return asum(self, axis)


def asarray(x: Any) -> Array:
    """Convenience constructor that mirrors `numpy.asarray`."""
    if isinstance(x, Array):
        return x
    return Array(x)


def ensure_array(x: Any) -> Array:
    """Like `asarray` but explicit about its intent in primitive wrappers."""
    return x if isinstance(x, Array) else Array(x)
