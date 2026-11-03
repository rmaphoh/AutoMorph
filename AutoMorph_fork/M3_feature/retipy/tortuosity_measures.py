# Retipy - Retinal Image Processing on Python
# Copyright (C) 2018  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Modified 2026-11-02 for AutoMorph_fork: the tortuosity measures are rewritten to follow
# Hart 1999 and Grisan 2008, widths include every centerline point, and the vessel equivalents
# return NaN instead of -1. See README.md for the full list of changes.

"""血管の中心線から蛇行度、血管幅、FD、血管密度、CRAE/CRVE を計算する。

蛇行度は、順に並べた中心線の点に平滑化スプラインを当てて計算する。画素の階段をそのまま微分すると、
距離法は斜めの向きで高く出て、曲率は階段の雑音で埋もれるためである。Grisan 2003, 2008 も平滑化スプラインを用いる。
"""

import numpy as np
import cv2
from scipy.interpolate import splev, splprep

from retipy.retina import Retina, detect_vessel_border

# 値は合成画像だけで決めた（README.md）。発症との関連を見る前に固定している
SMOOTHING = 0.35       # 点あたりの残差の二乗和の許容量（画素²）。x, y で共通なので向きによらない。直線に偽の曲率が出なくなる最小の候補
HYSTERESIS = 0.01      # 曲率の符号の変化とみなすしきい値（1/画素）。Grisan 2003 の静脈の値
MIN_PIXELS = 15        # AutoMorph の retipy.config の PixelsPerWindow（retipy 本家は 8）。これを超える区間だけを蛇行度と CRAE/CRVE に使う
SAMPLES_PER_PIXEL = 2  # スプラインの標本の密度


def fractal_dimension(Z):
    """AutoMorph と同じボックスカウント（元は retipy の lib/fractal_dimension.py）。全部が血管で埋まった箱は数えない"""
    assert len(Z.shape) == 2

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])

    p = min(Z.shape)
    n = 2 ** np.floor(np.log(p) / np.log(2))
    n = int(np.log(n) / np.log(2))
    sizes = 2 ** np.arange(n, 1, -1)
    counts = [boxcount(Z, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def vessel_density(Z):
    assert len(Z.shape) == 2
    return np.sum(Z == 1) / (Z.shape[0] * Z.shape[1])


class Centerline:
    """順に並べた画素に平滑化スプラインを当てた曲線。s は弧長、kappa は符号つきの曲率"""

    def __init__(self, x, y, smoothing=None):
        pts = np.column_stack([x, y]).astype(float)
        n = len(pts)
        if n <= 3:
            raise ValueError("スプラインには4点以上が必要")
        factor = SMOOTHING if smoothing is None else smoothing
        tck, _ = splprep([pts[:, 0], pts[:, 1]], k=3, s=factor * n)
        chain = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        t = np.linspace(0, 1, max(int(round(chain * SAMPLES_PER_PIXEL)), 8))
        self.xy = np.column_stack(splev(t, tck))
        d1 = np.column_stack(splev(t, tck, der=1))
        d2 = np.column_stack(splev(t, tck, der=2))
        speed = np.hypot(d1[:, 0], d1[:, 1])
        self.kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed, 1e-9) ** 3
        steps = np.hypot(*np.diff(self.xy, axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(steps)])

    @property
    def length(self):
        return float(self.s[-1])

    @property
    def chord(self):
        return float(np.hypot(*(self.xy[-1] - self.xy[0])))


def distance_measure_tortuosity(line: Centerline):
    """弧長 ÷ 弦長（Hart 1999 の τ1 に 1 を足した値）"""
    return line.length / line.chord


def squared_curvature_tortuosity(line: Centerline):
    """total squared curvature: κ² を弧長で積分した値。Hart 1999 の τ3 は媒介変数 t での ∫κ²dt で、弧長で書くのは Grisan 2003 の式(15)"""
    return float(np.trapezoid(line.kappa ** 2, line.s))


def turn_curve_cuts(line: Centerline, hysteresis=None):
    """曲率の符号が変わる位置を返す。しきい値を超えた区間の間で符号が変わったら、その中点で分ける（Grisan 2008）"""
    h = HYSTERESIS if hysteresis is None else hysteresis
    state = np.where(line.kappa > h, 1, np.where(line.kappa < -h, -1, 0))
    active = np.flatnonzero(state)
    cuts = []
    for a, b in zip(active[:-1], active[1:]):
        if state[a] != state[b]:
            cuts.append((a + b) // 2)
    return cuts


def tortuosity_density(line: Centerline, hysteresis=None):
    """Grisan 2008 の式(19): (n-1)/n × (1/L) × Σ(部分の弧長/部分の弦長 − 1)。n は曲率の符号が一定の部分の数"""
    cuts = turn_curve_cuts(line, hysteresis)
    n = len(cuts) + 1
    if n == 1:
        return 0.0
    edges = [0] + cuts + [len(line.s) - 1]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        chord = float(np.hypot(*(line.xy[b] - line.xy[a])))
        if chord > 0:
            total += (line.s[b] - line.s[a]) / chord - 1
    return (n - 1) / n * total / line.length


def Knudtson_cal(w1, w2):
    w_artery = 0.88 * np.sqrt(np.square(w1) + np.square(w2))
    w_vein = 0.95 * np.sqrt(np.square(w1) + np.square(w2))
    return w_artery, w_vein


def knudtson_trunk(widths, kind):
    """Knudtson 2003 の手順。太い順に6本を、細い×太いの組で1本になるまでまとめる。6本に満たなければ NaN"""
    w = sorted(widths)[-6:]
    if len(w) < 6:
        return np.nan
    idx = 0 if kind == "artery" else 1
    first = sorted(Knudtson_cal(w[i], w[5 - i])[idx] for i in range(3))
    second = sorted([Knudtson_cal(first[0], first[2])[idx], first[1]])
    return float(Knudtson_cal(second[0], second[1])[idx])


def _build_disc_offsets(max_r=80):
    size = 2 * max_r + 1
    canvas = np.zeros((size, size), dtype=np.uint8)
    out = [None]
    for r in range(1, max_r + 1):
        cv2.circle(canvas, (max_r, max_r), radius=r, color=255, thickness=-1)
        rr, cc = np.nonzero(canvas)
        out.append((rr - max_r, cc - max_r))
    return out


_DISC_OFFSETS = _build_disc_offsets()


def width_measurement(x, y, vessel_map):
    """AutoMorph と同じ円の方法で各点の幅を求める。元のコードは最後の点を落としていた（range(0, len(x) - 1)）"""
    h, w = vessel_map.shape
    out = []
    for i in range(len(x)):
        r, ok, vals = 0, True, None
        while ok:
            r += 1
            dr, dc = _DISC_OFFSETS[r]
            rr, cc = x[i] + dr, y[i] + dc
            keep = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            vals = vessel_map[rr[keep], cc[keep]]
            ok = bool(np.all(vals > 0))
        zeros = int((vals == 0).sum())
        out.append(r * 2 - 1 if zeros in (2, 3) else r * 2)
    return out


def segments_of(skeleton):
    """骨格から、分岐点を除いて順に並べた区間を取り出す"""
    r = Retina.__new__(Retina)
    r.np_image = (skeleton > 0).astype(np.float64)
    r.shape = r.np_image.shape
    return detect_vessel_border(r)


def skeleton_length(skeleton, segments):
    """骨格の実際の長さを推定する。分岐点で消した画素も含めるため、区間で求めた1画素あたりの長さを骨格全体に当てる"""
    pixels, length = 0, 0.0
    for vx, vy in segments:
        pixels += len(vx)
        length += Centerline(vx, vy).length + 1 if len(vx) > 3 else float(len(vx))
    if pixels == 0:
        return np.nan
    return float((skeleton > 0).sum()) * length / pixels


def evaluate_whole(mask, skeleton, min_pixels=MIN_PIXELS):
    """画像全体の FD、血管密度、血管幅、蛇行度3種。mask と skeleton は 912×912 の二値画像"""
    mask, skeleton = mask > 0, skeleton > 0
    segments = segments_of(skeleton)
    total_length = skeleton_length(skeleton, segments)
    lines = [Centerline(vx, vy) for vx, vy in segments if len(vx) > min_pixels]
    mean = lambda v: float(np.mean(v)) if len(v) else np.nan
    return {
        "fd": float(fractal_dimension(mask.astype(np.float64))),
        "vd": float(vessel_density(mask.astype(np.int64))),
        "width_pixelcount": float(mask.sum() / skeleton.sum()) if skeleton.sum() else np.nan,
        "width": float(mask.sum() / total_length) if total_length else np.nan,
        "dist": mean([distance_measure_tortuosity(c) for c in lines]),
        "tsc": mean([squared_curvature_tortuosity(c) for c in lines]),
        "mean_abs_curv": mean([float(np.trapezoid(np.abs(c.kappa), c.s)) / c.length for c in lines]),
        "dens": mean([tortuosity_density(c) for c in lines]),
        "n_seg": len(lines),
    }


def evaluate_zone(full_mask, zone_skeleton, kind, min_pixels=MIN_PIXELS):
    """Zone B, C の CRAE/CRVE（Knudtson）。値は 912×912 の画像上の画素単位。

    元のコードは円環で切ったマスクで幅を測るため、円環の縁で幅が小さく出る。ここでは切る前のマスクで測る。
    """
    vessel_map = (full_mask > 0).astype(np.uint8)
    widths = [float(np.mean(width_measurement(vx, vy, vessel_map)))
              for vx, vy in segments_of(zone_skeleton) if len(vx) > min_pixels]
    return {"cre": knudtson_trunk(widths, kind), "n_seg": len(widths)}
