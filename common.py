"""전압, 용량, 에너지밀도 계산 공통 함수."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Composition, Element, Lattice, Structure

import config

FARADAY_MAH = 26801.4  # mAh/mol (F / 3.6)


def na_bcc() -> Structure:
    """Na 금속 기준상 (bcc, a≈4.23 Å). M3GNet 으로 다시 최적화해서 μ_Na 로 사용."""
    return Structure(Lattice.cubic(4.23), ["Na", "Na"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def allowed(elements) -> bool:
    return not ({str(e) for e in elements} & config.BANNED_ELEMENTS)


def framework_and_x(comp: Composition) -> tuple[Composition, float]:
    """Na_x(F) 형태로 분해: 골격 F 의 '환산식 1개'당 Na 개수 x."""
    comp = Composition(comp)
    na = comp["Na"] if Element("Na") in comp else 0.0
    fw = Composition({el: amt for el, amt in comp.items() if el.symbol != "Na"})
    fw_reduced, factor = fw.get_reduced_composition_and_factor()
    return fw_reduced, na / factor


def per_framework(energy: float, volume: float, comp: Composition) -> tuple[float, float, float]:
    """전체 셀 에너지/부피를 골격 환산식 1개당 값으로 변환. (e, v, x) 반환."""
    comp = Composition(comp)
    fw = Composition({el: amt for el, amt in comp.items() if el.symbol != "Na"})
    _, factor = fw.get_reduced_composition_and_factor()
    _, x = framework_and_x(comp)
    return energy / factor, volume / factor, x


def lower_hull(xs: np.ndarray, ys: np.ndarray) -> list[int]:
    """(x, y) 점들의 아래쪽 볼록 껍질 인덱스 (x 오름차순)."""
    order = np.lexsort((ys, xs))
    hull: list[int] = []
    for i in order:
        while len(hull) >= 2:
            x1, y1 = xs[hull[-2]], ys[hull[-2]]
            x2, y2 = xs[hull[-1]], ys[hull[-1]]
            # 마지막 점이 (hull[-2] -> i) 선분보다 위에 있으면 제거
            if (x2 - x1) * (ys[i] - y1) - (y2 - y1) * (xs[i] - x1) <= 1e-12:
                hull.pop()
            else:
                break
        if not hull or xs[i] > xs[hull[-1]] + 1e-9:
            hull.append(i)
    return hull


def voltage_profile(points: list[dict], mu_na: float) -> dict:
    """같은 골격의 Na_x(F) 상들로 전압 곡선 계산.

    points: [{"x": Na/골격, "e": 골격당 에너지, "v": 골격당 부피, "id": ...}, ...]
    x 가 가장 작은 상(탈소듐화, 충전 상태)을 기준으로 생성에너지를 구하고,
    아래쪽 볼록 껍질을 따라 계단형 전압을 얻는다.  V_i = -ΔE_f/Δx  (1 e⁻ 반응, eV -> V)
    """
    xs = np.array([p["x"] for p in points], dtype=float)
    es = np.array([p["e"] for p in points], dtype=float)
    i0 = int(np.argmin(xs))
    ef = es - es[i0] - (xs - xs[i0]) * mu_na
    hull = lower_hull(xs, ef)
    hull = [h for h in hull if xs[h] >= xs[i0]]
    steps = []
    for a, b in zip(hull[:-1], hull[1:]):
        v = -(ef[b] - ef[a]) / (xs[b] - xs[a])
        steps.append({"x_from": xs[a], "x_to": xs[b], "V": float(v),
                      "id_charge": points[a]["id"], "id_discharge": points[b]["id"]})
    # 0 V 아래 계단은 Na 금속 석출이 먼저 일어나므로 사용 불가 -> 잘라냄.
    # 볼록 껍질이므로 전압은 x 증가에 따라 단조 감소 -> 앞에서부터 연속된 양(+) 구간만 사용
    usable = []
    for s in steps:
        if s["V"] <= 0.0:
            break
        usable.append(s)
    if not usable:
        return {"steps": steps, "usable": [], "avg_V": np.nan, "dx": 0.0}
    x_lo, x_hi = usable[0]["x_from"], usable[-1]["x_to"]
    dx = x_hi - x_lo
    avg_v = sum(s["V"] * (s["x_to"] - s["x_from"]) for s in usable) / dx
    return {"steps": steps, "usable": usable, "avg_V": float(avg_v), "dx": float(dx),
            "x_lo": float(x_lo), "x_hi": float(x_hi)}


def capacity_mAh_g(dx: float, framework: Composition, x_lo: float = 0.0) -> float:
    """탈소듐화(충전) 상태 질량 기준 용량. 하드카본(탄소 질량 기준)과 같은 기준."""
    mass = Composition(framework).weight + x_lo * Element("Na").atomic_mass
    return float(dx * FARADAY_MAH / mass)


def energy_density_proxy(v_anode: float, q_anode: float,
                         v_cat: float = None, q_cat: float = None) -> float:
    """활물질 기준 셀 에너지밀도 (Wh/kg, 양극+음극 활물질 질량 합 기준).

    E = (V_c - V_a) * Q_c*Q_a/(Q_c+Q_a)   (N/P = 1 가정)
    """
    v_cat = config.CATHODE["voltage"] if v_cat is None else v_cat
    q_cat = config.CATHODE["capacity"] if q_cat is None else q_cat
    if not np.isfinite(v_anode) or q_anode <= 0:
        return np.nan
    return float((v_cat - v_anode) * q_cat * q_anode / (q_cat + q_anode))


HC_ENERGY = energy_density_proxy(config.HARD_CARBON["voltage"], config.HARD_CARBON["capacity"])
