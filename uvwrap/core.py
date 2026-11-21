"""uvwrap 核心 API：UV 环绕（含导数变换）与各向异性 mip 选择。

本模块全部函数均为纯函数：不读取、不写入任何模块级可变状态，
也不缓存调用间数据，因此天然无状态且可被多线程并发安全调用。

环绕模式：

* ``repeat`` —— 取小数部分（周期 1），导数保持不变；
* ``clamp``  —— 限幅到 [0, 1]，区间外输出为常量、导数归零；
* ``mirror`` —— 周期 2 的三角波镜像，导数在镜像带内符号翻转、模长不变。

导数一律由环绕函数的*解析斜率*给出，而不是有限差分，因此跨整数
接缝时 repeat 的导数连续无跳变；mirror 在整数点的符号翻转是三角波
本身的内禀尖点（两侧斜率 +1/-1），并非数值跳变。
"""

import math
from collections import namedtuple

__all__ = [
    "MODE_REPEAT",
    "MODE_CLAMP",
    "MODE_MIRROR",
    "UvWrapError",
    "InvalidWrapModeError",
    "NonFiniteInputError",
    "InvalidTextureSizeError",
    "MipLevel",
    "wrap",
    "wrap_uv_derivatives",
    "mip_level",
    "UvWrapper",
]

MODE_REPEAT = "repeat"
MODE_CLAMP = "clamp"
MODE_MIRROR = "mirror"
_VALID_MODES = frozenset((MODE_REPEAT, MODE_CLAMP, MODE_MIRROR))


class UvWrapError(ValueError):
    """uvwrap 参数类错误的公共基类（同时也是 ValueError）。"""


class InvalidWrapModeError(UvWrapError):
    """传入了不支持的环绕模式枚举值。"""


class NonFiniteInputError(UvWrapError):
    """输入数值为 NaN 或 ±Inf 等非有限值。"""


class InvalidTextureSizeError(UvWrapError):
    """纹理尺寸为 0 或负数。"""


#: 三线性 mip 选择结果：(低层级, 高层级, 低层级权重, 高层级权重)。
#: 两个权重之和恒为 1；当钳制到端点时两个层级相同、权重为 (1, 0)。
MipLevel = namedtuple(
    "MipLevel", ("level_lo", "level_hi", "weight_lo", "weight_hi")
)


def _check_finite(name, value):
    if not math.isfinite(value):
        raise NonFiniteInputError(
            "%s must be a finite real number, got %r" % (name, value)
        )


def _wrap1(u, mode):
    """单轴环绕。

    返回 ``(环绕后的坐标, 环绕函数在 u 处的解析斜率)``。
    调用方需保证 ``u`` 有限且 ``mode`` 合法。
    """
    if mode == MODE_REPEAT:
        # Python 的 % 结果与除数同号，负坐标也能正确映射到 [0, 1)。
        return u % 1.0, 1.0
    if mode == MODE_CLAMP:
        if u < 0.0:
            return 0.0, 0.0
        if u > 1.0:
            return 1.0, 0.0
        return u, 1.0
    # mirror：周期 2 的三角波；floor(u) 的奇偶决定当前带内的斜率方向。
    f = math.floor(u)
    t = u - f
    if f % 2.0 != 0.0:
        return 1.0 - t, -1.0
    return t, 1.0


def wrap(u, mode=MODE_REPEAT):
    """对单个 UV 坐标做环绕，返回 [0, 1)（clamp 为 [0, 1]）内的 float。

    非法模式抛 :class:`InvalidWrapModeError`；非有限输入抛
    :class:`NonFiniteInputError`。
    """
    if mode not in _VALID_MODES:
        raise InvalidWrapModeError("unknown wrap mode: %r" % (mode,))
    _check_finite("u", u)
    return _wrap1(float(u), mode)[0]


def wrap_uv_derivatives(u, v, du, dv, mode=MODE_REPEAT):
    """环绕 (u, v) 并同步变换屏幕空间导数 (du, dv)。

    返回 4 元组 ``(u_wrapped, v_wrapped, du_wrapped, dv_wrapped)``：

    * repeat —— 导数原样保持，跨整数接缝无跳变；
    * mirror —— 处于镜像带时导数符号翻转、模长不变；
    * clamp  —— 区间内导数保持，区间外导数归零。

    纯函数，无状态，可并发调用。
    """
    if mode not in _VALID_MODES:
        raise InvalidWrapModeError("unknown wrap mode: %r" % (mode,))
    _check_finite("u", u)
    _check_finite("v", v)
    _check_finite("du", du)
    _check_finite("dv", dv)
    wu, su = _wrap1(float(u), mode)
    wv, sv = _wrap1(float(v), mode)
    return wu, wv, du * su, dv * sv


def mip_level(du, dv, tex_w, tex_h, levels):
    """各向异性 mip 层级选择与三线性插值权重。

    拉伸比按两个轴各自的纹素足迹 ``|du|*tex_w``、``|dv|*tex_h``
    的**最大值**取 ``log2``（取最大值而非平均，故各向异性拉伸不会
    退回各向同性），再钳制到 ``[0, levels-1]``（即 [0, levels)）。

    返回 :class:`MipLevel` —— ``(level_lo, level_hi, weight_lo,
    weight_hi)``，权重和恒为 1。

    非有限输入抛 :class:`NonFiniteInputError`；``tex_w``/``tex_h``
    为 0 或负数抛 :class:`InvalidTextureSizeError`；``levels < 1``
    抛 ``ValueError``。
    """
    _check_finite("du", du)
    _check_finite("dv", dv)
    _check_finite("tex_w", tex_w)
    _check_finite("tex_h", tex_h)
    _check_finite("levels", levels)
    if tex_w <= 0.0 or tex_h <= 0.0:
        raise InvalidTextureSizeError(
            "texture dimensions must be positive, got tex_w=%r tex_h=%r"
            % (tex_w, tex_h)
        )
    if levels < 1:
        raise ValueError("levels must be >= 1, got %r" % (levels,))

    stretch = max(abs(du) * tex_w, abs(dv) * tex_h)
    if stretch > 0.0:
        lam = math.log2(stretch)
    else:
        # 零足迹：理论层级为 -inf，钳制后即第 0 层。
        lam = -math.inf

    top = levels - 1
    lam_c = lam
    if lam_c < 0.0:
        lam_c = 0.0
    if lam_c > top:
        lam_c = float(top)

    lo = int(math.floor(lam_c))
    w_hi = lam_c - lo
    if w_hi == 0.0:
        # 精确落在某一层（含被钳制的端点）：无需跨层混合。
        hi = lo
        w_hi = 0.0
    else:
        hi = lo + 1
        if hi > top:  # 理论上不会发生，仅作防御
            hi = lo
            w_hi = 0.0
    return MipLevel(lo, hi, 1.0 - w_hi, w_hi)


class UvWrapper:
    """无状态环绕器；仅为兼容既有 API 而保留，方法为纯函数转发。"""

    def wrap(self, u, mode=MODE_REPEAT):
        """环绕单个坐标，等价于模块级 :func:`wrap`。"""
        return wrap(u, mode)
