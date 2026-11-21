"""新增行为测试：导数环绕、各向异性 mip、错误区分、并发、超大坐标。

文件名刻意包含 wrap，使 `pytest -k wrap` 能选中本模块。
"""

import math
import threading

import pytest

import uvwrap
from uvwrap import (
    InvalidTextureSizeError,
    InvalidWrapModeError,
    MipLevel,
    NonFiniteInputError,
    mip_level,
    wrap,
    wrap_uv_derivatives,
)


# ---------------------------------------------------------------- repeat

def test_wrap_repeat_basic_and_seams():
    assert wrap(0.0) == 0.0
    assert wrap(1.0) == 0.0
    assert wrap(2.0) == 0.0
    assert math.isclose(wrap(-0.25), 0.75, abs_tol=1e-12)
    assert math.isclose(wrap(0.5), 0.5, abs_tol=1e-12)
    assert 0.0 <= wrap(1000000.7) < 1.0


def test_wrap_repeat_derivatives_hold_across_seam():
    # 接缝两侧导数保持不变，不出现跳变。
    u0, _, d0, _ = wrap_uv_derivatives(0.999999, 0.3, 0.4, -0.5, "repeat")
    u1, _, d1, _ = wrap_uv_derivatives(1.000001, 0.3, 0.4, -0.5, "repeat")
    assert d0 == pytest.approx(0.4)
    assert d1 == pytest.approx(0.4)
    # 坐标在接缝两侧分别接近 1 与 0，几何上连续；导数完全相等。
    assert u0 > 0.99999
    assert u1 < 1e-5
    assert d0 == d1


def test_wrap_repeat_negative_derivative_preserved():
    wu, wv, du, dv = wrap_uv_derivatives(-1.7, -0.2, -0.6, 0.8, "repeat")
    assert math.isclose(wu, 0.3, abs_tol=1e-12)
    assert math.isclose(wv, 0.8, abs_tol=1e-12)
    assert du == pytest.approx(-0.6)
    assert dv == pytest.approx(0.8)


def test_wrap_repeat_huge_coordinates():
    # 超大坐标：结果仍落在 [0, 1)，且 repeat 导数保持不变。
    wu, _, du, _ = wrap_uv_derivatives(1e16 + 0.6, 0.0, 0.37, 0.0, "repeat")
    assert 0.0 <= wu < 1.0
    assert du == pytest.approx(0.37)
    assert 0.0 <= wrap(1e16) < 1.0
    assert math.isclose(wrap(1000.5), 0.5, abs_tol=1e-9)


# ---------------------------------------------------------------- mirror

def test_wrap_mirror_values():
    assert math.isclose(wrap(1.25, "mirror"), 0.75, abs_tol=1e-12)
    assert math.isclose(wrap(-0.25, "mirror"), 0.25, abs_tol=1e-12)
    assert wrap(2.0, "mirror") == 0.0
    assert wrap(3.0, "mirror") == 1.0
    assert math.isclose(wrap(2.3, "mirror"), 0.3, abs_tol=1e-12)
    assert math.isclose(wrap(-1.25, "mirror"), 0.75, abs_tol=1e-12)


def test_wrap_mirror_derivative_sign_flip():
    # 正向带 [0,1) 斜率 +1，镜像带 [1,2) 斜率 -1，模长不变。
    _, _, d_pos, _ = wrap_uv_derivatives(0.4, 0.0, 0.5, 0.0, "mirror")
    _, _, d_neg, _ = wrap_uv_derivatives(1.4, 0.0, 0.5, 0.0, "mirror")
    assert d_pos == pytest.approx(0.5)
    assert d_neg == pytest.approx(-0.5)
    # 负坐标同理：[-1,0) 是镜像带。
    _, _, d_n, _ = wrap_uv_derivatives(-0.6, 0.0, 0.5, 0.0, "mirror")
    assert d_n == pytest.approx(-0.5)


def test_wrap_mirror_seam_consistency():
    # 整数尖点两侧斜率模长一致、符号相反；取带内解析斜率，无虚假跳变。
    _, _, d_below, _ = wrap_uv_derivatives(0.999999, 0.0, 1.0, 0.0, "mirror")
    _, _, d_above, _ = wrap_uv_derivatives(1.000001, 0.0, 1.0, 0.0, "mirror")
    assert abs(d_below) == pytest.approx(1.0)
    assert abs(d_above) == pytest.approx(1.0)
    assert d_below == pytest.approx(-d_above)


# ---------------------------------------------------------------- clamp

def test_wrap_clamp_values_and_derivatives():
    assert wrap(2.0, "clamp") == 1.0
    assert wrap(-1.0, "clamp") == 0.0
    wu, _, du_in, _ = wrap_uv_derivatives(0.5, 0.0, 0.9, 0.0, "clamp")
    assert wu == pytest.approx(0.5)
    assert du_in == pytest.approx(0.9)
    _, _, du_hi, _ = wrap_uv_derivatives(1.5, 0.0, 0.9, 0.0, "clamp")
    assert du_hi == 0.0
    _, _, du_lo, _ = wrap_uv_derivatives(-0.5, 0.0, 0.9, 0.0, "clamp")
    assert du_lo == 0.0


# ---------------------------------------------------------------- mip

def test_mip_level_zero_footprint():
    r = mip_level(0.0, 0.0, 256, 256, 9)
    assert isinstance(r, MipLevel)
    assert r.level_lo == 0 and r.level_hi == 0
    assert r.weight_lo == pytest.approx(1.0)
    assert r.weight_hi == pytest.approx(0.0)


def test_mip_level_isotropic_exact_level():
    # 足迹宽 4 纹素 -> log2(4) = 2.0，权重全部落在第 2 层。
    r = mip_level(4.0 / 256, 4.0 / 256, 256, 256, 9)
    assert r.level_lo == 2 and r.level_hi == 2
    assert r.weight_lo == pytest.approx(1.0)
    assert r.weight_hi == pytest.approx(0.0)


def test_mip_level_anisotropic_does_not_collapse_to_isotropic():
    # du 足迹 1 纹素、dv 足迹 16 纹素；必须按最大轴选层，而非平均/最小值。
    r = mip_level(1.0 / 256, 16.0 / 256, 256, 256, 9)
    assert r.level_lo == 4
    # 交换两轴结果应相同（取 max 与轴顺序无关）。
    r2 = mip_level(16.0 / 256, 1.0 / 256, 256, 256, 9)
    assert tuple(r) == tuple(r2)


def test_mip_level_trilinear_weights():
    # 足迹 2^2.5 ≈ 5.657 -> 层级 2.5，权重各半。
    r = mip_level(2.0 ** 2.5 / 128, 0.0, 128, 128, 8)
    assert r.level_lo == 2 and r.level_hi == 3
    assert r.weight_lo == pytest.approx(0.5)
    assert r.weight_hi == pytest.approx(0.5)
    assert r.weight_lo + r.weight_hi == pytest.approx(1.0)


def test_mip_level_clamped_to_range():
    # 超大足迹钳制到 levels-1。
    r = mip_level(1e6, 1e6, 1024, 1024, 11)
    assert 0 <= r.level_lo < 11
    assert 0 <= r.level_hi < 11
    assert r.level_lo == 10 and r.level_hi == 10
    assert r.weight_hi == pytest.approx(0.0)
    # 单级纹理永远只有第 0 层。
    r0 = mip_level(100.0, 100.0, 8, 8, 1)
    assert tuple(r0) == (0, 0, 1.0, 0.0)


def test_mip_level_weights_always_sum_to_one():
    for k in range(1, 40):
        r = mip_level(float(k) / 64, float(k) / 128, 64, 128, 7)
        assert r.level_hi in (r.level_lo, r.level_lo + 1)
        assert r.weight_lo + r.weight_hi == pytest.approx(1.0)
        assert 0.0 <= r.weight_lo <= 1.0
        assert 0.0 <= r.weight_hi <= 1.0


# ---------------------------------------------------------------- errors

@pytest.mark.parametrize("fn", [
    lambda: wrap(0.5, "bogus"),
    lambda: wrap_uv_derivatives(0.0, 0.0, 0.1, 0.1, "bogus"),
])
def test_invalid_mode_distinguishable(fn):
    with pytest.raises(InvalidWrapModeError):
        fn()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_inputs_distinguishable(bad):
    with pytest.raises(NonFiniteInputError):
        wrap(bad)
    with pytest.raises(NonFiniteInputError):
        wrap_uv_derivatives(bad, 0.0, 0.1, 0.1)
    with pytest.raises(NonFiniteInputError):
        wrap_uv_derivatives(0.0, bad, 0.1, 0.1)
    with pytest.raises(NonFiniteInputError):
        mip_level(bad, 0.0, 8, 8, 4)


def test_zero_texture_size_distinguishable():
    with pytest.raises(InvalidTextureSizeError):
        mip_level(0.1, 0.1, 0, 8, 4)
    with pytest.raises(InvalidTextureSizeError):
        mip_level(0.1, 0.1, 8, 0, 4)
    with pytest.raises(InvalidTextureSizeError):
        mip_level(0.1, 0.1, -4, 8, 4)


def test_error_classes_are_value_errors():
    for exc in (InvalidWrapModeError, NonFiniteInputError,
                InvalidTextureSizeError):
        assert issubclass(exc, ValueError)


def test_bad_levels_value_error():
    with pytest.raises(ValueError):
        mip_level(0.1, 0.1, 8, 8, 0)


# ---------------------------------------------------------------- concurrency / stateless

def test_concurrent_calls_are_safe():
    errors = []

    def worker(seed):
        try:
            for i in range(2000):
                x = (i + seed) * 0.37
                w1 = wrap(x, "repeat")
                w2, _, d1, _ = wrap_uv_derivatives(x, -x, 0.25, -0.5,
                                                    "mirror")
                r = mip_level(0.01 * (i % 7), 0.02, 256, 128, 9)
                assert 0.0 <= w1 < 1.0
                assert 0.0 <= w2 <= 1.0
                assert 0.0 <= r.level_lo < 9
        except BaseException as exc:  # pragma: no cover - 失败路径
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_wrapper_matches_module_function():
    w = uvwrap.UvWrapper()
    assert w.wrap(0.1) == wrap(0.1)
    assert w.wrap(1.25, "mirror") == wrap(1.25, "mirror")
    # 无状态：连续调用互不影响。
    w.wrap(0.2)
    assert w.wrap(0.3) == 0.3
    assert isinstance(w.wrap(0.1), float)
