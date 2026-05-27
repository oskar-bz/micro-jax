"""Tiny MLP on XOR, trained with `jit(grad(...))`.

A 2 → 8 → 1 network with sigmoid activations, mean-squared loss, and plain
SGD. Each parameter has its own jit'd gradient function. Because `jit`
caches by shape signature, the trace happens exactly once per parameter,
and every subsequent training step reuses the compiled program.
"""

from __future__ import annotations

import numpy as np

import micro_jax as mjx
from micro_jax import Array


def sigmoid(x: Array) -> Array:
    one = Array(1.0)
    return one / (one + mjx.exp(-x))


def forward(W1: Array, b1: Array, W2: Array, b2: Array, X: Array) -> Array:
    h = sigmoid(mjx.matmul(X, W1) + b1)
    return sigmoid(mjx.matmul(h, W2) + b2)


def loss(W1: Array, b1: Array, W2: Array, b2: Array, X: Array, Y: Array) -> Array:
    pred = forward(W1, b1, W2, b2, X)
    err = pred - Y
    return mjx.sum(err * err) * Array(np.float32(1.0 / 4.0))


def main() -> None:
    # XOR truth table.
    X = Array(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    Y = Array(np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32))

    rng = np.random.RandomState(0)
    W1 = Array((rng.randn(2, 8) * 0.7).astype(np.float32))
    b1 = Array(np.zeros(8, dtype=np.float32))
    W2 = Array((rng.randn(8, 1) * 0.7).astype(np.float32))
    b2 = Array(np.zeros(1, dtype=np.float32))

    # One jit'd gradient function per parameter. They share the same trace
    # body but differ in argnum, so each gets its own compiled program.
    g_W1 = mjx.jit(mjx.grad(loss, 0))
    g_b1 = mjx.jit(mjx.grad(loss, 1))
    g_W2 = mjx.jit(mjx.grad(loss, 2))
    g_b2 = mjx.jit(mjx.grad(loss, 3))

    lr = Array(np.float32(2.0))  # XOR converges fast with a bold LR
    steps = 4000

    print("step  | loss")
    print("------+-------")
    for step in range(steps + 1):
        if step % 500 == 0:
            print(f"{step:>5} | {float(loss(W1, b1, W2, b2, X, Y)):.5f}")
        W1 = W1 - lr * g_W1(W1, b1, W2, b2, X, Y)
        b1 = b1 - lr * g_b1(W1, b1, W2, b2, X, Y)
        W2 = W2 - lr * g_W2(W1, b1, W2, b2, X, Y)
        b2 = b2 - lr * g_b2(W1, b1, W2, b2, X, Y)

    pred = forward(W1, b1, W2, b2, X).data.reshape(-1)
    print("\nfinal predictions vs. target:")
    print("  x       target  pred")
    for (x0, x1), y, p in zip(X.data, Y.data.reshape(-1), pred):
        print(f"  ({x0:.0f},{x1:.0f})    {y:.0f}     {p:.4f}")


if __name__ == "__main__":
    main()
