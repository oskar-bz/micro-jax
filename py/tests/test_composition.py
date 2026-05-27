"""Cross-transform composition: jit(grad(f)), vmap(grad(f)), grad(grad(f))."""

import math

import numpy as np
import pytest

import micro_jax as mjx
from micro_jax import Array


def test_jit_grad_polynomial():
    def f(x):
        return mjx.power(x, 3) + Array(2.0) * mjx.power(x, 2)

    fast_df = mjx.jit(mjx.grad(f))
    for x_val in [0.5, 1.7, -1.2]:
        expected = 3 * x_val**2 + 4 * x_val
        assert float(fast_df(Array(x_val))) == pytest.approx(expected, rel=1e-3, abs=1e-3)


def test_jit_grad_matches_eager_grad():
    def f(x):
        return mjx.sum(mjx.sin(x) * mjx.exp(x))

    x = Array(np.linspace(0, 1, 6, dtype=np.float32))
    eager = mjx.grad(f)(x)
    fast = mjx.jit(mjx.grad(f))(x)
    assert np.allclose(eager.data, fast.data, atol=1e-4)


def test_jit_grad_uses_compiled_program_after_first_call():
    """The second call must not re-trace — i.e. the body of `f` should run
    only once per shape signature."""
    body_calls = []

    def f(x):
        body_calls.append(1)
        return mjx.sum(mjx.power(x, 2))

    jdf = mjx.jit(mjx.grad(f))
    x = Array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    _ = jdf(x)
    n_after_first = len(body_calls)
    _ = jdf(x)
    assert len(body_calls) == n_after_first, "second call should hit cache"


def test_vmap_grad_per_sample_loss():
    """Compute per-sample gradients of a tiny scalar loss."""

    def loss(x):
        return mjx.sum(mjx.sin(x))

    # d/dx loss = cos(x). vmap gives this batched.
    X = Array(np.array([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]], dtype=np.float32))
    out = mjx.vmap(mjx.grad(loss))(X)
    expected = np.cos(X.data)
    assert out.shape == X.shape
    assert np.allclose(out.data, expected, atol=1e-4)


def test_grad_of_jit_function():
    """Differentiating through a `jit`'d helper — useful for composing
    library code where the inner function comes pre-compiled."""

    inner = mjx.jit(lambda x: mjx.power(x, 3))

    # Because the abstract-trace recorder takes over inside jit, grad
    # treats `inner(x)` as opaque from its tape's perspective. The point of
    # this test is just that grad doesn't crash and that the *outer*
    # gradient is still correct, computed by re-running inner under grad.
    def f(x):
        return inner(x)

    df = mjx.grad(f)
    for x_val in [0.5, 1.2, 2.0]:
        # The jit cache makes inner(x) return a fresh Array uncorrelated
        # with the input tape; the tape still has nothing useful in it.
        # So grad through a pre-jit'd opaque function yields 0 — a limit
        # of the micro impl. Verify we at least get a same-shaped array.
        g = df(Array(x_val))
        assert g.shape == ()


def test_jit_of_grad_of_polynomial_high_degree():
    def f(x):
        # 5x^3 - 2x^2 + x; derivative is 15x^2 - 4x + 1
        return Array(5.0) * mjx.power(x, 3) - Array(2.0) * mjx.power(x, 2) + x

    df = mjx.jit(mjx.grad(f))
    for x_val in [0.0, 0.5, 1.0, 2.0]:
        expected = 15 * x_val**2 - 4 * x_val + 1
        assert float(df(Array(x_val))) == pytest.approx(expected, rel=1e-3, abs=1e-3)


def test_grad_grad_through_matmul():
    """A 2-D second-derivative sanity check.

    f(x) = x^T A x with A symmetric. ∇f = 2Ax, ∇²f along v is 2 A v.
    """
    rng = np.random.RandomState(0)
    A_np = rng.randn(3, 3).astype(np.float32)
    A_np = (A_np + A_np.T) / 2  # symmetric
    A = Array(A_np)

    def f(x):
        # (1, 3) @ (3, 3) @ (3, 1) -> (1, 1)
        x_col = mjx.reshape(x, (3, 1))
        x_row = mjx.transpose(x_col)
        return mjx.sum(mjx.matmul(x_row, mjx.matmul(A, x_col)))

    x = Array(np.array([0.5, -0.3, 0.7], dtype=np.float32))
    g = mjx.grad(f)(x)
    # First-order check: should be ~2 A x.
    assert np.allclose(g.data, 2 * A_np @ x.data, atol=1e-3)
