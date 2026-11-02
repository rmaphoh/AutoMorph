"""AutoMorph_fork の修正の単体テスト。曲線と幅について、真の値が分かる入力で確かめる。

  python AutoMorph_fork/tests/test_fork.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "M3_feature"))
from retipy import math as m, tortuosity_measures as tm  # noqa: E402
from retipy.retina import order_vessel_pixels  # noqa: E402


def smooth_curve(radius: float, half_angle: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(-half_angle, half_angle, n)
    return 300 + radius * np.sin(t), 300 + radius * np.cos(t)


def test_second_difference_has_no_extra_divisor():
    assert m.derivative2_centered_h1(1, [0, 1, 4]) == 2  # y = x² の2階微分
    for bad in (0, 2):
        try:
            m.derivative2_centered_h1(bad, [0, 1, 4])
        except ValueError:
            continue
        raise AssertionError("端の点で例外にならない")


def test_order_from_middle_start():
    path = [(10, 10 + i) for i in range(11)]
    start_mid = [path[5], path[4], path[6], path[3], path[7], path[2], path[8], path[1], path[9], path[0], path[10]]
    ordered = order_vessel_pixels(start_mid)
    assert ordered in ([list(p) for p in path], [list(p) for p in reversed(path)])


def test_order_rejects_ring():
    ring = [(0, 1), (1, 2), (2, 1), (1, 0)]
    assert order_vessel_pixels(ring) is None


def test_arc_values_match_theory():
    radius, half = 200.0, np.deg2rad(45)
    c = tm.Centerline(*smooth_curve(radius, half))
    arc = 2 * half * radius
    # 許容幅は平滑化スプラインの実測の誤差に合わせた（弧長 +0.3%、TSC +9%、弧長÷弦長 +0.004）
    assert abs(c.length - arc) / arc < 5e-3
    assert abs(tm.squared_curvature_tortuosity(c) - arc / radius ** 2) / (arc / radius ** 2) < 0.10  # ∫κ²ds = L/R²
    assert abs(tm.distance_measure_tortuosity(c) - half / np.sin(half)) < 5e-3
    assert tm.tortuosity_density(c) == 0.0  # 曲率の符号が一定なら0（Grisan 2008）


def test_squared_curvature_does_not_cancel_on_s_curve():
    x = np.linspace(0, 400, 800)
    y = 30 * np.sin(2 * np.pi * x / 200)
    c = tm.Centerline(x, y)
    assert tm.squared_curvature_tortuosity(c) > 0.05
    assert len(tm.turn_curve_cuts(c)) == 3
    assert tm.tortuosity_density(c) > 0


def test_scale_response():
    a = tm.Centerline(*smooth_curve(200.0, np.deg2rad(45)))
    b = tm.Centerline(*smooth_curve(400.0, np.deg2rad(45), n=800))
    # 弧長で積分するので2倍にすると 1/2。Hart 1999 の表1の τ3 は媒介変数 t での積分で 1/γ²
    assert abs(tm.squared_curvature_tortuosity(a) / tm.squared_curvature_tortuosity(b) - 2) < 0.1
    assert abs(tm.distance_measure_tortuosity(a) - tm.distance_measure_tortuosity(b)) < 1e-3


def test_rotation_invariance():
    x, y = smooth_curve(150.0, np.deg2rad(60))
    base = tm.Centerline(x, y)
    a = np.deg2rad(37)
    rx, ry = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    rot = tm.Centerline(rx, ry)
    assert abs(tm.distance_measure_tortuosity(base) - tm.distance_measure_tortuosity(rot)) < 1e-4
    assert abs(tm.squared_curvature_tortuosity(base) / tm.squared_curvature_tortuosity(rot) - 1) < 0.01


def test_knudtson_worked_example():
    # Knudtson 2003 の計算例: 100, 90, 80, 70, 60, 50 µm → CRAE 134.4 µm
    assert abs(tm.knudtson_trunk([100, 90, 80, 70, 60, 50], "artery") - 134.4) < 0.1
    assert np.isnan(tm.knudtson_trunk([100, 90], "artery"))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
