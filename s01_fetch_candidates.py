"""1단계: 후보 구조 수집 -> data/candidates.json

트랙 A (삽입형): Materials Project 의 Na 삽입 전극 DB에서 '충전/방전 구조 쌍'을 가져온다.
            DFT 평균전압은 '넓은 1차 필터'와 '나중에 M3GNet 정확도 검증'에만 쓴다.
            전압 자체는 2단계에서 M3GNet 으로 다시 계산한다.
트랙 B (합금/화합물형): Na-M 이원계(M = Sn, Sb, P, ...)의 모든 (준)안정상을 가져와
            2단계에서 M3GNet 에너지로 볼록 껍질을 만들고 전압 곡선을 계산한다.

사용법:
    export MP_API_KEY=...        # https://next-gen.materialsproject.org/api
    python s01_fetch_candidates.py
    python s01_fetch_candidates.py --demo   # API 키 없이 파이프라인 테스트용 예제 3종
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pymatgen.core import Lattice, Structure

import config
from common import allowed

DATA = Path(__file__).parent / "data"


def fetch_mp() -> dict:
    from mp_api.client import MPRester

    key = os.environ.get("MP_API_KEY")
    if not key:
        raise SystemExit("MP_API_KEY 환경변수를 설정하세요 (또는 --demo).")

    out = {"insertion": [], "alloy": [], "structures": {}}
    need_ids: set[str] = set()

    with MPRester(key) as mpr:
        # ---------- 트랙 A ----------
        docs = mpr.materials.insertion_electrodes.search(
            working_ion="Na",
            average_voltage=(0.0, config.MP_MAX_DFT_VOLTAGE),
            fields=[
                "framework_formula", "average_voltage", "capacity_grav",
                "max_delta_volume", "stability_charge", "stability_discharge",
                "id_charge", "id_discharge", "material_ids", "elements",
            ],
        )
        n_el = n_stab = 0
        for d in docs:
            els = [str(e) for e in d.elements]
            if not allowed(els):
                continue
            n_el += 1
            if max(d.stability_charge or 0, d.stability_discharge or 0) > config.MP_MAX_EHULL:
                continue
            n_stab += 1
            ids = {str(d.id_charge), str(d.id_discharge)} | {str(m) for m in d.material_ids or []}
            out["insertion"].append({
                "battery_id": f"{d.framework_formula}_{d.id_charge}_{d.id_discharge}",
                "framework": str(d.framework_formula),
                "ids": sorted(ids),
                "dft_avg_V": d.average_voltage,
                "dft_capacity": d.capacity_grav,
                "dft_dV": d.max_delta_volume,
            })
            need_ids |= ids
        print(f"[트랙 A] Na 삽입 전극 {len(out['insertion'])}개")

        # ---------- 트랙 B ----------
        for m in config.ALLOY_HOSTS:
            if not allowed([m]):
                continue
            sdocs = mpr.materials.summary.search(
                chemsys=[m, f"Na-{m}"],
                energy_above_hull=(0, config.MP_MAX_EHULL),
                fields=["material_id", "formula_pretty", "nsites", "energy_above_hull"],
            )
            ids = sorted(str(s.material_id) for s in sdocs if s.nsites <= config.MAX_SITES)
            if len(ids) >= 2:
                out["alloy"].append({"host": m, "ids": ids})
                need_ids |= set(ids)
        print(f"[트랙 B] Na-M 이원계 {len(out['alloy'])}개")

        # ---------- 구조 다운로드 ----------
        need = sorted(need_ids)
        for i in range(0, len(need), 500):
            for s in mpr.materials.summary.search(
                material_ids=need[i:i + 500],
                fields=["material_id", "structure", "energy_above_hull", "formula_pretty"],
            ):
                out["structures"][str(s.material_id)] = {
                    "formula": s.formula_pretty,
                    "e_hull_dft": s.energy_above_hull,
                    "structure": s.structure.as_dict(),
                }

    # 원자 수가 너무 큰 구조가 들어간 전극은 제외
    def small(mid):
        st = out["structures"].get(mid)
        return st is not None and len(st["structure"]["sites"]) <= config.MAX_SITES

    out["insertion"] = [e for e in out["insertion"] if all(small(i) for i in e["ids"])]
    out["alloy"] = [a for a in out["alloy"] if all(small(i) for i in a["ids"])]
    out["funnel_na"] = {f"Na 전극 (V<{config.MP_MAX_DFT_VOLTAGE} vs Na)": len(docs),
                        "원소 제한": n_el, "안정성 (E_hull≤0.1)": n_stab,
                        "원자 수 ≤ 60": len(out["insertion"])}
    print(f"원자 수 필터 후: 삽입형 {len(out['insertion'])}, 합금형 {len(out['alloy'])}, "
          f"구조 {len(out['structures'])}개")
    return out


def demo() -> dict:
    """API 키 없이 돌려보는 예제 (과학적 결론용 아님)."""
    tis2 = Structure(Lattice.hexagonal(3.41, 5.70), ["Ti", "S", "S"],
                     [[0, 0, 0], [1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]])
    natis2 = Structure(Lattice.hexagonal(3.45, 6.90), ["Ti", "S", "S", "Na"],
                       [[0, 0, 0], [1 / 3, 2 / 3, 0.22], [2 / 3, 1 / 3, 0.78], [0, 0, 0.5]])
    sb = Structure(Lattice.hexagonal(4.31, 11.27), ["Sb"] * 6,
                   [[0, 0, 0.2336], [0, 0, 0.7664],
                    [2 / 3, 1 / 3, 0.5669], [2 / 3, 1 / 3, 0.0997],
                    [1 / 3, 2 / 3, 0.9003], [1 / 3, 2 / 3, 0.4331]])
    na3sb = Structure.from_spacegroup("Fm-3m", Lattice.cubic(7.42), ["Sb", "Na", "Na"],
                                      [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]])
    na3sb = na3sb.get_primitive_structure()
    st = {"demo-TiS2": tis2, "demo-NaTiS2": natis2, "demo-Sb": sb, "demo-Na3Sb": na3sb}
    return {
        "insertion": [{"battery_id": "demo-TiS2_Na", "framework": "TiS2",
                       "ids": ["demo-TiS2", "demo-NaTiS2"],
                       "dft_avg_V": None, "dft_capacity": None, "dft_dV": None}],
        "alloy": [{"host": "Sb", "ids": ["demo-Sb", "demo-Na3Sb"]}],
        "structures": {k: {"formula": v.composition.reduced_formula, "e_hull_dft": None,
                           "structure": v.as_dict()} for k, v in st.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    data = demo() if args.demo else fetch_mp()
    (DATA / "candidates.json").write_text(json.dumps(data))
    print(f"저장: {DATA / 'candidates.json'}")


if __name__ == "__main__":
    main()
