"""Linear regression: closed-form vs. gradient descent with `grad`.

The dataset is `y = 2x + 1 + noise`. We fit `w, b` two ways:

1. The textbook closed-form (using numpy directly) — as a ground truth.
2. Pure gradient descent on the MSE loss, with gradients computed by
   `micro_jax.grad`. We `jit` the gradient functions for free speed-up.
"""

from __future__ import annotations

import numpy as np

import micro_jax as mjx
from micro_jax import Array


def main() -> None:
    rng = np.random.RandomState(0)
    N = 50
    xs_np = np.linspace(-2.0, 2.0, N).astype(np.float32)
    ys_np = (2.0 * xs_np + 1.0 + 0.3 * rng.randn(N)).astype(np.float32)

    # --- 1. closed-form reference ---------------------------------------
    w_cf = float(np.cov(xs_np, ys_np, bias=True)[0, 1] / np.var(xs_np))
    b_cf = float(ys_np.mean() - w_cf * xs_np.mean())
    print(f"closed-form: w={w_cf:+.4f}  b={b_cf:+.4f}")

    # --- 2. gradient descent --------------------------------------------
    xs = Array(xs_np)
    ys = Array(ys_np)
    inv_N = Array(np.float32(1.0 / N))

    def loss(w: Array, b: Array, xs: Array, ys: Array) -> Array:
        err = w * xs + b - ys
        return mjx.sum(err * err) * inv_N

    grad_w = mjx.jit(mjx.grad(loss, argnum=0))
    grad_b = mjx.jit(mjx.grad(loss, argnum=1))

    w = Array(0.0)
    b = Array(0.0)
    lr = Array(0.05)

    print("\nstep |    w       b      loss")
    print("-----+----------------------------")
    for step in range(201):
        if step % 25 == 0:
            print(f"{step:>4} | {float(w):+.4f} {float(b):+.4f}  {float(loss(w, b, xs, ys)):.4f}")
        w = w - lr * grad_w(w, b, xs, ys)
        b = b - lr * grad_b(w, b, xs, ys)

    print(f"\nGD final:    w={float(w):+.4f}  b={float(b):+.4f}")
    print(f"|dw|={abs(float(w) - w_cf):.2e}  |db|={abs(float(b) - b_cf):.2e}")


if __name__ == "__main__":
    main()
