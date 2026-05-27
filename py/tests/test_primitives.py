"""Forward-pass sanity checks for every primitive against numpy."""

import numpy as np
import pytest

import micro_jax as mjx
from micro_jax import Array


def to_np(x):
    return x.data if isinstance(x, Array) else np.asarray(x, dtype=np.float32)


def test_add_simple():
    a = Array([1.0, 2.0, 3.0])
    b = Array([10.0, 20.0, 30.0])
    out = mjx.add(a, b)
    assert np.allclose(out.data, [11.0, 22.0, 33.0])


def test_add_broadcasts():
    a = Array(np.arange(6, dtype=np.float32).reshape(2, 3))
    b = Array(np.arange(3, dtype=np.float32))
    out = a + b
    assert out.shape == (2, 3)
    assert np.allclose(out.data, a.data + b.data)


def test_mul_and_neg_and_sub():
    a = Array([1.0, 2.0, 3.0])
    b = Array([4.0, 5.0, 6.0])
    assert np.allclose((a * b).data, [4.0, 10.0, 18.0])
    assert np.allclose((-a).data, [-1.0, -2.0, -3.0])
    assert np.allclose((a - b).data, [-3.0, -3.0, -3.0])


def test_div_and_reciprocal():
    a = Array([1.0, 2.0, 4.0])
    b = Array([2.0, 2.0, 2.0])
    assert np.allclose((a / b).data, [0.5, 1.0, 2.0])
    assert np.allclose(mjx.reciprocal(a).data, [1.0, 0.5, 0.25])


def test_pow():
    a = Array([1.0, 2.0, 3.0])
    assert np.allclose(mjx.power(a, 2).data, [1.0, 4.0, 9.0])
    assert np.allclose(mjx.power(a, 0.5).data, np.sqrt([1.0, 2.0, 3.0]))


def test_trig_exp_log():
    x = Array([0.5, 1.0, 1.5])
    assert np.allclose(mjx.sin(x).data, np.sin(x.data), atol=1e-6)
    assert np.allclose(mjx.cos(x).data, np.cos(x.data), atol=1e-6)
    assert np.allclose(mjx.exp(x).data, np.exp(x.data), atol=1e-5)
    assert np.allclose(mjx.log(x).data, np.log(x.data), atol=1e-6)


def test_matmul_2d():
    a = Array(np.arange(6, dtype=np.float32).reshape(2, 3))
    b = Array(np.arange(12, dtype=np.float32).reshape(3, 4))
    out = mjx.matmul(a, b)
    assert out.shape == (2, 4)
    assert np.allclose(out.data, a.data @ b.data)


def test_sum_full_and_axes():
    x = Array(np.arange(12, dtype=np.float32).reshape(3, 4))
    assert np.allclose(mjx.sum(x).data, x.data.sum())
    assert np.allclose(mjx.sum(x, axis=0).data, x.data.sum(axis=0))
    assert np.allclose(mjx.sum(x, axis=1).data, x.data.sum(axis=1))


def test_reshape_and_transpose():
    x = Array(np.arange(6, dtype=np.float32))
    r = mjx.reshape(x, (2, 3))
    assert r.shape == (2, 3)
    t = mjx.transpose(r)
    assert t.shape == (3, 2)
    assert np.allclose(t.data, x.data.reshape(2, 3).T)


def test_array_size_one_to_float():
    x = Array([42.0])
    assert float(x) == pytest.approx(42.0)


def test_array_rejects_float_for_non_scalar():
    x = Array([1.0, 2.0])
    with pytest.raises(TypeError):
        float(x)
