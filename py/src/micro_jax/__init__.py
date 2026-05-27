"""micro-jax — a tiny educational reimplementation of JAX.

What `micrograd` is to PyTorch, this is to JAX:

* a single `Array` type that wraps a numpy buffer,
* a registry of primitives with `fwd`/`vjp`/`batch_rule`/`abstract_eval`,
* three composable transforms: `grad`, `jit`, `vmap`.

See `spec.md` (alongside this package) for the language- and
implementation-agnostic specification this code implements.
"""

from __future__ import annotations

from .array import Array, asarray
from .grad import grad, value_and_grad
from .jit import jit
from .ops import (
    add,
    asum,
    broadcast_to,
    cos,
    div,
    exp,
    log,
    matmul,
    mul,
    neg,
    ones_like,
    power,
    reciprocal,
    reshape,
    sin,
    sub,
    sum_to_shape,
    transpose,
    zeros_like,
)
from .primitives import REGISTRY, Primitive
from .vmap import vmap


# Provide `sum` under its idiomatic name without shadowing the builtin in
# this module's namespace. Users do `mjx.sum(x, axis=0)`.
sum = asum


__all__ = [
    "Array",
    "Primitive",
    "REGISTRY",
    "asarray",
    "add",
    "asum",
    "broadcast_to",
    "cos",
    "div",
    "exp",
    "grad",
    "jit",
    "log",
    "matmul",
    "mul",
    "neg",
    "ones_like",
    "power",
    "reciprocal",
    "reshape",
    "sin",
    "sub",
    "sum",
    "sum_to_shape",
    "transpose",
    "value_and_grad",
    "vmap",
    "zeros_like",
]
