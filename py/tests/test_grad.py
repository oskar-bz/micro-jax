"""Tests for `grad`, including higher-order and broadcasting cases."""

import math

import numpy as np
import pytest

import micro_jax as mjx
from micro_jax import Array


def test_grad_polynomial_scalar():
    # f(x) = x^3 + 2x^2 + 1, f'(x) = 3x^2 + 4x
    def f(x):
        return mjx.power(x, 3) + 2.0 * mjx.power(x, 2) + Array(1.0)

    df = mjx.grad(f)
    for x_val in [0.0, 1.0, -1.5, 2.7]:
        g = df(Array(x_val))
        expected = 3 * x_val**2 + 4 * x_val
        assert float(g) == pytest.approx(expected, rel=1e-4, abs=1e-5)


def test_grad_trig():
    # f(x) = sin(x) * cos(x); f' = cos(x)^2 - sin(x)^2 = cos(2x)
    def f(x):
        return mjx.sin(x) * mjx.cos(x)

    df = mjx.grad(f)
    for x_val in [0.3, 1.1, -0.7]:
        g = df(Array(x_val))
        assert float(g) == pytest.approx(math.cos(2 * x_val), abs=1e-5)


def test_grad_exp_log():
    # f(x) = log(exp(x)) = x; f' = 1
    def f(x):
        return mjx.log(mjx.exp(x))

    for x_val in [0.5, 1.2, 2.0]:
        assert float(mjx.grad(f)(Array(x_val))) == pytest.approx(1.0, abs=1e-4)


def test_grad_returns_correct_shape():
    # f(x) = sum(x^2); gradient should be 2*x.
    def f(x):
        return mjx.sum(mjx.power(x, 2))

    x = Array([1.0, 2.0, 3.0, 4.0])
    g = mjx.grad(f)(x)
    assert g.shape == (4,)
    assert np.allclose(g.data, 2 * x.data)


def test_grad_through_broadcast():
    # f(x, b) = sum(x + b); ∂f/∂x = ones_like(x), ∂f/∂b = N
    def f(x, b):
        return mjx.sum(x + b)

    x = Array(np.ones((3, 4), dtype=np.float32))
    b = Array(np.zeros((4,), dtype=np.float32))

    gx = mjx.grad(f, argnum=0)(x, b)
    gb = mjx.grad(f, argnum=1)(x, b)

    assert gx.shape == x.shape
    assert gb.shape == b.shape
    assert np.allclose(gx.data, np.ones_like(x.data))
    # `b` is broadcast across 3 rows; each entry contributes 3 times.
    assert np.allclose(gb.data, 3 * np.ones_like(b.data))


def test_grad_matmul():
    # f(A) = sum(A @ B). dA = (sum-grad broadcast to (m,n)) @ B^T = ones @ B^T.
    A = Array(np.random.RandomState(0).randn(3, 4).astype(np.float32))
    B = Array(np.random.RandomState(1).randn(4, 5).astype(np.float32))

    def f(A):
        return mjx.sum(mjx.matmul(A, B))

    gA = mjx.grad(f)(A)
    expected = np.ones((3, 5), dtype=np.float32) @ B.data.T
    assert np.allclose(gA.data, expected, atol=1e-5)


def test_grad_argnum():
    def f(x, y):
        return x * y + mjx.sin(y)

    gx = mjx.grad(f, argnum=0)(Array(2.0), Array(1.0))
    gy = mjx.grad(f, argnum=1)(Array(2.0), Array(1.0))
    assert float(gx) == pytest.approx(1.0, abs=1e-5)
    assert float(gy) == pytest.approx(2.0 + math.cos(1.0), abs=1e-5)


def test_grad_of_grad_polynomial():
    # f(x) = x^4; f'(x) = 4x^3; f''(x) = 12x^2.
    def f(x):
        return mjx.power(x, 4)

    d2f = mjx.grad(mjx.grad(f))
    for x_val in [0.5, 1.0, 2.0]:
        out = d2f(Array(x_val))
        assert float(out) == pytest.approx(12 * x_val**2, rel=1e-3, abs=1e-3)


def test_grad_of_grad_sin():
    # f(x) = sin(x); f'' = -sin(x).
    f = mjx.sin
    d2f = mjx.grad(mjx.grad(f))
    for x_val in [0.0, 0.7, 1.3]:
        assert float(d2f(Array(x_val))) == pytest.approx(-math.sin(x_val), abs=1e-4)


def test_value_and_grad():
    def f(x):
        return mjx.power(x, 3)

    v, g = mjx.value_and_grad(f)(Array(2.0))
    assert float(v) == pytest.approx(8.0)
    assert float(g) == pytest.approx(12.0)


def test_grad_independent_argument_returns_zero():
    # f(x, y) = x * 2; gradient w.r.t. y must be zero.
    def f(x, y):
        return x * Array(2.0)

    g = mjx.grad(f, argnum=1)(Array(3.0), Array(5.0))
    assert float(g) == pytest.approx(0.0)


def test_grad_pure_function_can_be_called_multiple_times():
    def f(x):
        return mjx.power(x, 2)

    df = mjx.grad(f)
    # Calling repeatedly should produce identical results (no leaked state).
    a = df(Array(3.0))
    b = df(Array(3.0))
    assert float(a) == float(b) == pytest.approx(6.0)
