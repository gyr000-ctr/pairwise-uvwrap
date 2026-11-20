import math

import pytest

import uvwrap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _check_repeat_zero():
    return uvwrap.UvWrapper().wrap(0.0) == 0.0


def _check_repeat_half():
    return abs(uvwrap.UvWrapper().wrap(0.5) - 0.5) < 1e-12


def _check_repeat_one():
    return uvwrap.UvWrapper().wrap(1.0) == 0.0


def _check_repeat_neg():
    return abs(uvwrap.UvWrapper().wrap(-0.25) - 0.75) < 1e-12


def _check_clamp_high():
    return uvwrap.UvWrapper().wrap(2.0, "clamp") == 1.0


def _check_clamp_low():
    return uvwrap.UvWrapper().wrap(-1.0, "clamp") == 0.0


def _check_mirror():
    return abs(uvwrap.UvWrapper().wrap(1.25, "mirror") - 0.75) < 1e-12


def _check_mirror_neg():
    return abs(uvwrap.UvWrapper().wrap(-0.25, "mirror") - 0.25) < 1e-12


def _check_bad_mode():
    return _raises(lambda: uvwrap.UvWrapper().wrap(0.5, "bogus"))


def _check_det():
    return uvwrap.UvWrapper().wrap(3.3) == uvwrap.UvWrapper().wrap(3.3)


def _check_type():
    return isinstance(uvwrap.UvWrapper().wrap(0.1), float)


def _check_range():
    return 0.0 <= uvwrap.UvWrapper().wrap(7.7) < 1.0


def _check_int_u():
    return uvwrap.UvWrapper().wrap(2.0) == 0.0


def _check_large():
    return abs(uvwrap.UvWrapper().wrap(1000.5) - 0.5) < 1e-9


def _check_no_state():
    w = uvwrap.UvWrapper()
    w.wrap(0.2)
    return w.wrap(0.3) == 0.3

CASES = [("repeat_zero", "_check_repeat_zero"), ("repeat_half", "_check_repeat_half"), ("repeat_one", "_check_repeat_one"), ("repeat_neg", "_check_repeat_neg"), ("clamp_high", "_check_clamp_high"), ("clamp_low", "_check_clamp_low"), ("mirror", "_check_mirror"), ("mirror_neg", "_check_mirror_neg"), ("bad_mode", "_check_bad_mode"), ("det", "_check_det"), ("type", "_check_type"), ("range", "_check_range"), ("int_u", "_check_int_u"), ("large", "_check_large"), ("no_state", "_check_no_state")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
