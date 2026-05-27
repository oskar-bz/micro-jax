"""Tests for `jit` — correctness, caching, and the optimization passes."""

import numpy as np
import pytest

import micro_jax as mjx
from micro_jax import Array


def test_jit_matches_eager_simple():
    def f(x):
        return mjx.sin(x) * mjx.exp(x) + Array(2.0) * x

    jf = mjx.jit(f)
    for x_val in [0.5, 1.2, -0.8]:
        x = Array(x_val)
        assert np.allclose(jf(x).data, f(x).data, atol=1e-5)


def test_jit_cache_keyed_by_shape_signature():
    calls = []

    def f(x):
        calls.append(x.shape)
        return x * Array(3.0)

    jf = mjx.jit(f)
    jf(Array(np.ones((4,), dtype=np.float32)))
    jf(Array(np.ones((4,), dtype=np.float32)))  # cache hit
    jf(Array(np.ones((4, 4), dtype=np.float32)))  # cache miss — new shape
    # f's body only runs once per shape, during tracing.
    assert calls == [(4,), (4, 4)]


def test_jit_dead_code_elimination():
    """An entry whose output is unused must be removed."""

    def f(x):
        unused = mjx.sin(x) * mjx.exp(x)  # noqa: F841 — intentionally dead
        return x + x

    jf = mjx.jit(f)
    _ = jf(Array(1.0))
    program = next(iter(jf._cache.values()))
    prim_names = [e.prim.name for e in program.entries]
    # The dead sin/exp/mul chain should be gone; we expect only the add.
    assert "sin" not in prim_names
    assert "exp" not in prim_names


def test_jit_constant_folding():
    """Operations on closure-captured constants get folded away."""
    a = Array(2.0)
    b = Array(3.0)

    def f(x):
        # `a + b` does not depend on `x` — it should be folded.
        return x + (a + b)

    jf = mjx.jit(f)
    _ = jf(Array(1.0))
    program = next(iter(jf._cache.values()))
    prim_names = [e.prim.name for e in program.entries]
    # Only the dependent add should remain; the constant add is folded.
    assert prim_names.count("add") == 1


def test_jit_common_subexpression_elimination():
    """Two identical sub-expressions of the input collapse into one entry."""

    def f(x):
        a = mjx.sin(x)
        b = mjx.sin(x)
        return a + b

    jf = mjx.jit(f)
    _ = jf(Array(0.7))
    program = next(iter(jf._cache.values()))
    sin_count = sum(1 for e in program.entries if e.prim.name == "sin")
    assert sin_count == 1


def test_jit_closure_constants_baked_in():
    coef = Array(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def f(x):
        return mjx.sum(x * coef)

    jf = mjx.jit(f)
    x = Array(np.array([10.0, 20.0, 30.0], dtype=np.float32))
    assert float(jf(x)) == pytest.approx(140.0, abs=1e-4)


def test_jit_preserves_higher_dim_shapes():
    def f(A, B):
        return mjx.matmul(A, B)

    jf = mjx.jit(f)
    A = Array(np.random.RandomState(0).randn(3, 4).astype(np.float32))
    B = Array(np.random.RandomState(1).randn(4, 5).astype(np.float32))
    out = jf(A, B)
    assert out.shape == (3, 5)
    assert np.allclose(out.data, A.data @ B.data, atol=1e-4)


def test_jit_returning_input_unchanged():
    """Edge case: function returns its argument as-is."""

    def f(x):
        return x

    jf = mjx.jit(f)
    x = Array([1.0, 2.0, 3.0])
    out = jf(x)
    assert np.allclose(out.data, x.data)
