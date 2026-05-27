"""Tests for `vmap` — equivalence to an explicit python `for` loop."""

import numpy as np
import pytest

import micro_jax as mjx
from micro_jax import Array


def test_vmap_elementwise():
    def f(x):
        return mjx.sin(x) * mjx.exp(x)

    X = Array(np.linspace(0, 1, 5, dtype=np.float32))
    out = mjx.vmap(f)(X)
    expected = np.stack([f(Array(float(x))).data for x in X.data])
    assert out.shape == X.shape
    assert np.allclose(out.data, expected, atol=1e-5)


def test_vmap_two_args():
    def f(x, y):
        return x + y * Array(2.0)

    X = Array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    Y = Array(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    out = mjx.vmap(f)(X, Y)
    expected = X.data + 2 * Y.data
    assert np.allclose(out.data, expected)


def test_vmap_matmul_over_vectors():
    # f(v) = v @ A for fixed A: batched matmul.
    rng = np.random.RandomState(0)
    A = Array(rng.randn(3, 4).astype(np.float32))
    V = Array(rng.randn(5, 3).astype(np.float32))  # batch of row vectors

    def f(v):
        # Reshape into (1, 3) so the matmul primitive (which expects 2-D) works.
        v2d = mjx.reshape(v, (1, 3))
        out = mjx.matmul(v2d, A)
        return mjx.reshape(out, (4,))

    out = mjx.vmap(f)(V)
    expected = V.data @ A.data
    assert out.shape == (5, 4)
    assert np.allclose(out.data, expected, atol=1e-4)


def test_vmap_with_unbatched_input():
    # in_axes=(0, None): the first input is batched, the second is not.
    def f(x, b):
        return x + b

    X = Array(np.arange(8, dtype=np.float32).reshape(4, 2))
    b = Array(np.array([10.0, 20.0], dtype=np.float32))
    out = mjx.vmap(f, in_axes=(0, None))(X, b)
    assert out.shape == (4, 2)
    assert np.allclose(out.data, X.data + b.data)


def test_vmap_per_sample_grad():
    """The canonical composition test: vmap(grad(f)) → per-sample gradients."""

    def loss(x):
        # scalar loss = sum(x^2). dl/dx = 2x. Per-sample gradients should
        # be the inputs doubled, batched together.
        return mjx.sum(mjx.power(x, 2))

    per_sample_grad = mjx.vmap(mjx.grad(loss))
    X = Array(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
    out = per_sample_grad(X)
    expected = 2 * X.data
    assert out.shape == X.shape
    assert np.allclose(out.data, expected, atol=1e-5)


def test_vmap_rejects_mismatched_batch_sizes():
    X = Array(np.zeros((3, 2), dtype=np.float32))
    Y = Array(np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        mjx.vmap(lambda x, y: x + y)(X, Y)


def test_vmap_requires_at_least_one_batched_input():
    with pytest.raises(ValueError):
        mjx.vmap(lambda x, y: x + y, in_axes=(None, None))(Array(1.0), Array(2.0))


def test_vmap_broadcasts_non_dependent_output():
    """Function output that ignored the batched input should still produce a
    properly-shaped batched output."""

    const = Array(np.array([7.0, 8.0], dtype=np.float32))

    def f(_x):
        return const

    X = Array(np.zeros((4, 2), dtype=np.float32))
    out = mjx.vmap(f)(X)
    assert out.shape == (4, 2)
    assert np.allclose(out.data, np.broadcast_to(const.data, (4, 2)))
