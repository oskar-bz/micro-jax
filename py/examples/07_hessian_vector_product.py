"""Hessian-vector products without ever materializing the Hessian.

The trick is the identity

    H(f)(x) . v  =  d/dx <grad(f)(x), v>

i.e. take the gradient of f, dot it with v (so the result is scalar again),
and differentiate that scalar w.r.t. x. The outer `grad` differentiates
through the *operations of the inner `grad`'s reverse pass* -- only
possible because every VJP in micro-jax is written in terms of `Array`
operations and therefore re-traceable.

We check two cases:

* f(x) = 0.5 * ||x||^2     =>  H = I             =>  Hv = v.
* f(x) = sum sin(x)         =>  H = diag(-sin x)  =>  Hv = -sin(x) * v.
"""

from __future__ import annotations

import numpy as np

import micro_jax as mjx
from micro_jax import Array


def hvp(f, x: Array, v: Array) -> Array:
    """Hessian-vector product for a scalar-valued f."""

    def inner(x_inner: Array) -> Array:
        return mjx.sum(mjx.grad(f)(x_inner) * v)

    return mjx.grad(inner)(x)


def main() -> None:
    rng = np.random.RandomState(0)
    x = Array(rng.randn(4).astype(np.float32))
    v = Array(rng.randn(4).astype(np.float32))

    # --- Quadratic: f(x) = ½ ||x||²  =>  Hv = v ---------------------------
    def f_quad(x: Array) -> Array:
        return Array(0.5) * mjx.sum(x * x)

    hv = hvp(f_quad, x, v)
    print("Case 1: f(x) = 0.5 * ||x||^2")
    print(f"  v        = {v.data}")
    print(f"  Hv       = {hv.data}")
    print(f"  max err  = {np.max(np.abs(hv.data - v.data)):.2e}\n")

    # --- f(x) = sum sin(x)  =>  H = diag(-sin x), so Hv = -sin(x) * v ----
    def f_sin(x: Array) -> Array:
        return mjx.sum(mjx.sin(x))

    hv = hvp(f_sin, x, v)
    expected = -np.sin(x.data) * v.data
    print("Case 2: f(x) = sum sin(x)")
    print(f"  -sin(x)*v = {expected}")
    print(f"  Hv        = {hv.data}")
    print(f"  max err   = {np.max(np.abs(hv.data - expected)):.2e}")


if __name__ == "__main__":
    main()
