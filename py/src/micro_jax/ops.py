"""Built-in primitives and the user-facing operations that invoke them.

Each primitive is paired with a thin python function that handles input
coercion and (where it matters) broadcasting. The VJP rules are written
entirely in terms of these python functions — never raw numpy — so that
higher-order differentiation can re-trace them through `_emit`.

Broadcasting is handled by an explicit `broadcast_to` primitive in the
forward pass. Its VJP is a `sum_to_shape`, and `sum_to_shape`'s VJP is a
`broadcast_to`. That symmetry keeps the math honest: every shape change in
the forward pass has a matching shape change in the reverse pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .array import Array, ensure_array
from .primitives import Primitive


# ---------------------------------------------------------------------------
# Shape primitives — broadcast / reshape / transpose / sum.
# These show up implicitly in many VJP rules.
# ---------------------------------------------------------------------------


def _broadcast_to_fwd(x: np.ndarray, *, shape: tuple[int, ...]) -> np.ndarray:
    # np.broadcast_to returns a read-only view; copy to keep the buffer owned.
    return np.broadcast_to(x, shape).copy()


def _broadcast_to_vjp(g: Array, x: Array, *, shape: tuple[int, ...]) -> tuple[Array]:
    return (sum_to_shape(g, x.shape),)


def _broadcast_to_batch_rule(input_data, batch_dims, params):
    x = input_data[0]
    bd = batch_dims[0]
    shape = tuple(params["shape"])
    if bd is None:
        return np.broadcast_to(x, shape).copy(), None
    # Move the batch axis to position 0, then broadcast each batch element
    # to the requested *unbatched* shape — i.e., the result is
    # (batch, *shape). We may need to insert size-1 axes after the batch
    # so that numpy broadcasting aligns from the right.
    x = np.moveaxis(x, bd, 0)
    n = x.shape[0]
    target = (n,) + shape
    while x.ndim < len(target):
        x = np.expand_dims(x, 1)
    return np.broadcast_to(x, target).copy(), 0


broadcast_to_p = Primitive(
    "broadcast_to", _broadcast_to_fwd, _broadcast_to_vjp,
    batch_rule=_broadcast_to_batch_rule,
)


def broadcast_to(x: Any, shape: tuple[int, ...]) -> Array:
    x = ensure_array(x)
    shape = tuple(int(s) for s in shape)
    if x.shape == shape:
        return x
    return broadcast_to_p.bind(x, shape=shape)


def _reshape_fwd(x: np.ndarray, *, shape: tuple[int, ...]) -> np.ndarray:
    return np.reshape(x, shape)


def _reshape_vjp(g: Array, x: Array, *, shape: tuple[int, ...]) -> tuple[Array]:
    return (reshape(g, x.shape),)


def _reshape_batch_rule(input_data, batch_dims, params):
    x = input_data[0]
    bd = batch_dims[0]
    shape = tuple(params["shape"])
    if bd is None:
        return np.reshape(x, shape), None
    x = np.moveaxis(x, bd, 0)
    return np.reshape(x, (x.shape[0],) + shape), 0


reshape_p = Primitive(
    "reshape", _reshape_fwd, _reshape_vjp, batch_rule=_reshape_batch_rule
)


def reshape(x: Any, shape: tuple[int, ...]) -> Array:
    x = ensure_array(x)
    shape = tuple(int(s) for s in shape)
    if x.shape == shape:
        return x
    return reshape_p.bind(x, shape=shape)


def _transpose_fwd(x: np.ndarray, *, axes: tuple[int, ...] | None = None) -> np.ndarray:
    return np.transpose(x, axes)


def _transpose_vjp(g: Array, x: Array, *, axes: tuple[int, ...] | None = None) -> tuple[Array]:
    if axes is None:
        return (transpose(g),)
    # The inverse permutation undoes the forward transpose.
    inv = [0] * len(axes)
    for i, a in enumerate(axes):
        inv[a] = i
    return (transpose(g, axes=tuple(inv)),)


def _transpose_batch_rule(input_data, batch_dims, params):
    x = input_data[0]
    bd = batch_dims[0]
    axes = params.get("axes")
    if bd is None:
        return np.transpose(x, axes), None
    x = np.moveaxis(x, bd, 0)
    if axes is None:
        # Reverse all non-batch dims (mirrors `np.transpose` defaults but
        # keeps the batch axis pinned at the front).
        new_axes = (0,) + tuple(reversed(range(1, x.ndim)))
    else:
        # Each axis index referred to the unbatched view; shift past the
        # batch axis at position 0.
        new_axes = (0,) + tuple(a + 1 for a in axes)
    return np.transpose(x, new_axes), 0


transpose_p = Primitive(
    "transpose", _transpose_fwd, _transpose_vjp, batch_rule=_transpose_batch_rule
)


def transpose(x: Any, axes: tuple[int, ...] | None = None) -> Array:
    x = ensure_array(x)
    return transpose_p.bind(x, axes=axes)


def _sum_axes_fwd(
    x: np.ndarray,
    *,
    axes: tuple[int, ...] | None,
    keepdims: bool,
) -> np.ndarray:
    return np.sum(x, axis=axes, keepdims=keepdims)


def _sum_axes_vjp(
    g: Array,
    x: Array,
    *,
    axes: tuple[int, ...] | None,
    keepdims: bool,
) -> tuple[Array]:
    # The gradient is just `g` broadcast back to `x.shape`. The only twist is
    # that when `keepdims=False` we have to re-insert size-1 axes first so
    # that broadcasting lines up.
    if not keepdims:
        if axes is None:
            ax = tuple(range(x.ndim))
        else:
            ax = tuple(a % x.ndim for a in axes)
        gshape = list(g.shape)
        for a in sorted(ax):
            gshape.insert(a, 1)
        g = reshape(g, tuple(gshape))
    return (broadcast_to(g, x.shape),)


def _sum_axes_batch_rule(input_data, batch_dims, params):
    x = input_data[0]
    bd = batch_dims[0]
    axes = params["axes"]
    keepdims = params["keepdims"]
    if bd is None:
        return np.sum(x, axis=axes, keepdims=keepdims), None
    x = np.moveaxis(x, bd, 0)
    if axes is None:
        # "Sum everything" in unbatched view = sum every non-batch axis.
        new_axes = tuple(range(1, x.ndim))
    else:
        new_axes = tuple(a + 1 for a in axes)
    return np.sum(x, axis=new_axes, keepdims=keepdims), 0


sum_axes_p = Primitive(
    "sum_axes", _sum_axes_fwd, _sum_axes_vjp, batch_rule=_sum_axes_batch_rule
)


def asum(
    x: Any,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Sum across `axis`. Named `asum` to avoid clashing with builtin `sum`."""
    x = ensure_array(x)
    if axis is None:
        axes: tuple[int, ...] | None = None
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)
    return sum_axes_p.bind(x, axes=axes, keepdims=bool(keepdims))


def sum_to_shape(x: Array, target_shape: tuple[int, ...]) -> Array:
    """Reduce `x` along axes that were broadcast away to match `target_shape`.

    Inverse of `broadcast_to`. Used in the VJP of every broadcasting op.
    """
    target_shape = tuple(target_shape)
    if x.shape == target_shape:
        return x
    # Sum the extra leading axes that didn't exist in the target shape.
    n_extra = x.ndim - len(target_shape)
    if n_extra > 0:
        x = asum(x, axis=tuple(range(n_extra)), keepdims=False)
    # For each remaining axis where the target is 1 but x is larger, sum it
    # back down to 1.
    reduce_axes = tuple(
        i for i, (xs, ts) in enumerate(zip(x.shape, target_shape)) if ts == 1 and xs != 1
    )
    if reduce_axes:
        x = asum(x, axis=reduce_axes, keepdims=True)
    return x


# ---------------------------------------------------------------------------
# Elementwise primitives
# ---------------------------------------------------------------------------


def _binary_broadcast(x: Array, y: Array) -> tuple[Array, Array]:
    """Bring `x` and `y` to a common shape using numpy's broadcast rules.

    The actual broadcast is performed via the `broadcast_to` primitive so that
    the reverse pass automatically un-broadcasts gradients via its VJP.
    """
    shape = np.broadcast_shapes(x.shape, y.shape)
    if x.shape != shape:
        x = broadcast_to(x, shape)
    if y.shape != shape:
        y = broadcast_to(y, shape)
    return x, y


def _add_fwd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y


def _add_vjp(g: Array, x: Array, y: Array) -> tuple[Array, Array]:
    # Inputs are pre-broadcast to a common shape in the wrapper below, so the
    # gradient flows back unchanged on both sides.
    return g, g


add_p = Primitive("add", _add_fwd, _add_vjp)


def add(x: Any, y: Any) -> Array:
    x, y = _binary_broadcast(ensure_array(x), ensure_array(y))
    return add_p.bind(x, y)


def _mul_fwd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * y


def _mul_vjp(g: Array, x: Array, y: Array) -> tuple[Array, Array]:
    return g * y, g * x


mul_p = Primitive("mul", _mul_fwd, _mul_vjp)


def mul(x: Any, y: Any) -> Array:
    x, y = _binary_broadcast(ensure_array(x), ensure_array(y))
    return mul_p.bind(x, y)


def _neg_fwd(x: np.ndarray) -> np.ndarray:
    return -x


def _neg_vjp(g: Array, x: Array) -> tuple[Array]:
    return (-g,)


neg_p = Primitive("neg", _neg_fwd, _neg_vjp)


def neg(x: Any) -> Array:
    return neg_p.bind(ensure_array(x))


def sub(x: Any, y: Any) -> Array:
    return add(x, neg(y))


def _sin_fwd(x: np.ndarray) -> np.ndarray:
    return np.sin(x)


def _sin_vjp(g: Array, x: Array) -> tuple[Array]:
    return (g * cos(x),)


sin_p = Primitive("sin", _sin_fwd, _sin_vjp)


def sin(x: Any) -> Array:
    return sin_p.bind(ensure_array(x))


def _cos_fwd(x: np.ndarray) -> np.ndarray:
    return np.cos(x)


def _cos_vjp(g: Array, x: Array) -> tuple[Array]:
    return (-g * sin(x),)


cos_p = Primitive("cos", _cos_fwd, _cos_vjp)


def cos(x: Any) -> Array:
    return cos_p.bind(ensure_array(x))


def _exp_fwd(x: np.ndarray) -> np.ndarray:
    return np.exp(x)


def _exp_vjp(g: Array, x: Array) -> tuple[Array]:
    # d/dx exp(x) = exp(x); recomputing via `exp(x)` keeps the VJP traceable.
    return (g * exp(x),)


exp_p = Primitive("exp", _exp_fwd, _exp_vjp)


def exp(x: Any) -> Array:
    return exp_p.bind(ensure_array(x))


def _log_fwd(x: np.ndarray) -> np.ndarray:
    return np.log(x)


def _log_vjp(g: Array, x: Array) -> tuple[Array]:
    return (g / x,)


log_p = Primitive("log", _log_fwd, _log_vjp)


def log(x: Any) -> Array:
    return log_p.bind(ensure_array(x))


def _pow_fwd(x: np.ndarray, *, n: float) -> np.ndarray:
    return np.power(x, n).astype(np.float32)


def _pow_vjp(g: Array, x: Array, *, n: float) -> tuple[Array]:
    return (g * float(n) * power(x, n - 1),)


pow_p = Primitive("pow", _pow_fwd, _pow_vjp)


def power(x: Any, n: float | int) -> Array:
    return pow_p.bind(ensure_array(x), n=float(n))


def reciprocal(x: Any) -> Array:
    return power(x, -1.0)


def div(x: Any, y: Any) -> Array:
    return mul(x, reciprocal(y))


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


def _matmul_fwd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def _matmul_vjp(g: Array, a: Array, b: Array) -> tuple[Array, Array]:
    # 2-D case: dA = g @ B^T, dB = A^T @ g (see spec §2.3).
    # We restrict ourselves to 2-D inputs here; supporting fully general
    # broadcasting matmul is out of scope for the micro version.
    return matmul(g, transpose(b)), matmul(transpose(a), g)


matmul_p = Primitive("matmul", _matmul_fwd, _matmul_vjp)


def matmul(a: Any, b: Any) -> Array:
    return matmul_p.bind(ensure_array(a), ensure_array(b))


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def zeros_like(x: Array) -> Array:
    return Array(np.zeros_like(x.data))


def ones_like(x: Array) -> Array:
    return Array(np.ones_like(x.data))
