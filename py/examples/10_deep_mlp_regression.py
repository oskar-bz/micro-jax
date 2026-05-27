"""4-layer MLP fitting sin(x): eager vs. jit'd training, timed.

The point of this example is performance, not modeling. The network is
[1 -> 32 -> 32 -> 32 -> 1] with `tanh` activations and MSE loss. Each
parameter has its own `grad(loss, argnum=i)`; we then build a jit'd
version of each gradient function and compare per-step wall time.

`jit(grad(loss))` traces the entire forward + reverse pass into one
AbstractTape, then runs DCE -> const-fold -> dedup-constants -> CSE
-> DCE on it. The compiled program is just a flat list of numpy calls,
so each training step skips all the per-op `Array` allocation and tape
bookkeeping that eager mode pays for.
"""

from __future__ import annotations

import time

import numpy as np

import micro_jax as mjx
from micro_jax import Array


def tanh(x: Array) -> Array:
    ep = mjx.exp(x)
    en = mjx.exp(-x)
    return (ep - en) / (ep + en)


def init_params(rng: np.random.RandomState, dims: list[int]) -> list[Array]:
    params: list[Array] = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        scale = float(np.sqrt(2.0 / (d_in + d_out)))  # Xavier
        params.append(Array((rng.randn(d_in, d_out) * scale).astype(np.float32)))
        params.append(Array(np.zeros(d_out, dtype=np.float32)))
    return params


def forward(params: list[Array], X: Array) -> Array:
    h = X
    n_layers = len(params) // 2
    for i in range(n_layers):
        h = mjx.matmul(h, params[2 * i]) + params[2 * i + 1]
        if i < n_layers - 1:
            h = tanh(h)
    return h


def loss_fn(*args: Array) -> Array:
    # args = (W1, b1, W2, b2, ..., Wn, bn, X, Y)
    *params, X, Y = args
    pred = forward(list(params), X)
    err = pred - Y
    return mjx.sum(err * err) * Array(np.float32(1.0 / X.data.shape[0]))


def main() -> None:
    rng = np.random.RandomState(0)

    # --- dataset: 128 samples of y = sin(x) on [-pi, pi] ----------------
    N = 128
    X_np = np.linspace(-np.pi, np.pi, N).reshape(-1, 1).astype(np.float32)
    Y_np = np.sin(X_np).astype(np.float32)
    X = Array(X_np)
    Y = Array(Y_np)

    # --- model: 4-layer MLP, 32 hidden units per layer ------------------
    dims = [1, 32, 32, 32, 1]
    params_init = init_params(rng, dims)
    n_params = len(params_init)
    print(f"network: {dims}  ({n_params} parameter arrays)")

    # One gradient function per parameter — separate in eager and jit form.
    grads_eager = [mjx.grad(loss_fn, argnum=i) for i in range(n_params)]
    grads_jit = [mjx.jit(mjx.grad(loss_fn, argnum=i)) for i in range(n_params)]

    lr = Array(np.float32(0.05))
    initial_loss = float(loss_fn(*params_init, X, Y))
    print(f"initial loss: {initial_loss:.5f}\n")

    def sgd_step(grads, params):
        return [params[i] - lr * grads[i](*params, X, Y) for i in range(n_params)]

    # --- eager timing ---------------------------------------------------
    n_eager = 30
    params_eager = list(params_init)
    sgd_step(grads_eager, params_eager)  # warmup (jit-less, but caches JIT memory)
    t0 = time.perf_counter()
    for _ in range(n_eager):
        params_eager = sgd_step(grads_eager, params_eager)
    t_eager = time.perf_counter() - t0
    eager_ms = (t_eager / n_eager) * 1000
    print(f"eager:  {n_eager:4d} steps in {t_eager:6.3f}s  ({eager_ms:6.2f} ms/step)")

    # --- jit timing -----------------------------------------------------
    n_jit_warmup = 1
    n_jit = 500
    params_jit = list(params_init)
    for _ in range(n_jit_warmup):
        params_jit = sgd_step(grads_jit, params_jit)  # triggers tracing
    t0 = time.perf_counter()
    for _ in range(n_jit):
        params_jit = sgd_step(grads_jit, params_jit)
    t_jit = time.perf_counter() - t0
    jit_ms = (t_jit / n_jit) * 1000
    print(f"jit:    {n_jit:4d} steps in {t_jit:6.3f}s  ({jit_ms:6.2f} ms/step)")

    print(f"\nspeedup: {eager_ms / jit_ms:.1f}x")

    # --- a snapshot of one of the compiled programs ---------------------
    sample_prog = next(iter(grads_jit[0]._cache.values()))
    print(
        f"\ngrads_jit[0] compiled program: {len(sample_prog.entries)} ops, "
        f"{len(sample_prog.constants)} folded constants"
    )

    # --- long training run with jit so we can see convergence -----------
    print("\n--- training run (jit, 3000 steps) ---")
    params = list(params_init)
    params = sgd_step(grads_jit, params)  # warmup with re-used cache
    t0 = time.perf_counter()
    for s in range(3000):
        params = sgd_step(grads_jit, params)
        if s % 500 == 0:
            print(f"  step {s + 1:>4}: loss = {float(loss_fn(*params, X, Y)):.5f}")
    print(f"  step 3000: loss = {float(loss_fn(*params, X, Y)):.5f}")
    print(f"  total time: {time.perf_counter() - t0:.2f}s")

    # --- sanity-check the fit at a few points ---------------------------
    pred = forward(params, X).data.reshape(-1)
    actual = Y.data.reshape(-1)
    print("\nprediction samples:")
    print("    x          pred         target       |err|")
    for i in [0, 16, 32, 48, 64, 80, 96, 112, 127]:
        err = abs(pred[i] - actual[i])
        print(f"  {X.data[i, 0]:+.3f}     {pred[i]:+.5f}    {actual[i]:+.5f}    {err:.5f}")


if __name__ == "__main__":
    main()
