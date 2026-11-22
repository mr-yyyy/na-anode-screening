"""2단계: M3GNet(ASE) 구조 최적화 -> 전압/용량/부피변화/에너지밀도 -> data/stage2.csv

- 모든 구조 + Na 금속(bcc)을 ASE FIRE + FrechetCellFilter 로 원자 위치와 격자를 함께 최적화
- 삽입형: 같은 골격의 Na_xF 상들 -> 볼록 껍질 -> 계단 전압
- 합금형: Na-M 이원계 상들 -> 볼록 껍질 -> 계단 전압
- MP DFT 평균전압이 있으면 같이 기록 -> M3GNet 오차 분석에 사용
- 이미 최적화한 구조(data/relaxed/*.json)는 건너뜀 -> 중간에 끊겨도 이어서 실행 가능

사용법:  python s02_relax_voltage.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import config
from common import (HC_ENERGY, capacity_mAh_g, energy_density_proxy, framework_and_x,
                    na_bcc, per_framework, voltage_profile)
from mlip import get_calculator

DATA = Path(__file__).parent / "data"
RELAXED = DATA / "relaxed"


def relax_all(structs: dict[str, Structure]) -> dict:
    """{id: (relaxed Structure, energy, converged)}"""
    RELAXED.mkdir(parents=True, exist_ok=True)
    calc = None
    results = {}
    ids = sorted(structs, key=lambda k: len(structs[k]))
    for n, k in enumerate(ids, 1):
        f = RELAXED / f"{k}.json"
        if f.exists():
            d = json.loads(f.read_text())
            results[k] = (Structure.from_dict(d["structure"]), d["energy"], d["converged"])
            continue
        calc = calc or get_calculator()
        atoms = AseAtomsAdaptor.get_atoms(structs[k])
        atoms.calc = calc
        t0 = time.time()
        opt = FIRE(FrechetCellFilter(atoms), logfile=None)
        conv = bool(opt.run(fmax=config.RELAX_FMAX, steps=config.RELAX_MAX_STEPS))
        e = float(atoms.get_potential_energy())
        s = AseAtomsAdaptor.get_structure(atoms)
        results[k] = (s, e, conv)
        f.write_text(json.dumps({"structure": s.as_dict(), "energy": e, "converged": conv}))
        print(f"  [{n}/{len(ids)}] {k:<20} {s.composition.reduced_formula:<12} "
              f"E/atom={e / len(s):8.4f}  steps={opt.nsteps:4d}  conv={conv}  {time.time() - t0:5.1f}s")
    return results


def main():
    data = json.loads((DATA / "candidates.json").read_text())
    structs = {k: Structure.from_dict(v["structure"]) for k, v in data["structures"].items()}
    structs["__Na_bcc__"] = na_bcc()
    print(f"구조 {len(structs)}개 최적화 시작 (모델: {config.M3GNET_MODEL})")

    res = relax_all(structs)

    na_s, na_e, _ = res.pop("__Na_bcc__")
    mu_na = na_e / len(na_s)
    print(f"μ_Na (M3GNet, bcc) = {mu_na:.4f} eV/atom")

    def pts(id_list):
        p = []
        for i in id_list:
            s, e, c = res[i]
            ef, vf, x = per_framework(e, s.volume, s.composition)
            p.append({"id": i, "e": ef, "v": vf, "x": x, "converged": c,
                      "formula": s.composition.reduced_formula})
        return p

    rows = []
    tracks = (("insertion", data["insertion"]), ("alloy", data["alloy"]),
              ("substituted", data.get("substituted", [])))
    for kind, groups in tracks:
        for g in groups:
            p = pts(g["ids"])
            fws = {framework_and_x(Composition(q["formula"]))[0].reduced_formula for q in p}
            if len(fws) != 1:  # 합금 트랙: 순수 Na 상(x=∞)은 골격이 없으므로 제외
                p = [q for q in p if q["formula"] != "Na"]
                fws = {framework_and_x(Composition(q["formula"]))[0].reduced_formula for q in p}
            if len(fws) != 1 or len(p) < 2:
                continue
            fw = framework_and_x(Composition(p[0]["formula"]))[0]
            prof = voltage_profile(p, mu_na)
            if not prof["usable"]:
                rows.append({"id": g.get("battery_id", f"Na-{g.get('host')}"), "kind": kind,
                             "framework": fw.reduced_formula, "avg_V": math.nan,
                             "note": "0 V 이상인 반응 없음"})
                continue
            # 같은 조성의 다형체가 여러 개일 수 있으므로 볼록 껍질 위의 상(id)으로 끝점을 고른다
            by_id = {q["id"]: q for q in p}
            lo = by_id[prof["usable"][0]["id_charge"]]
            hi = by_id[prof["usable"][-1]["id_discharge"]]
            q_cap = capacity_mAh_g(prof["dx"], fw, prof["x_lo"])
            dv = 100.0 * (hi["v"] - lo["v"]) / lo["v"]
            e_cell = energy_density_proxy(prof["avg_V"], q_cap)
            rows.append({
                "id": g.get("battery_id", f"Na-{g.get('host')}"),
                "kind": kind,
                "framework": fw.reduced_formula,
                "charged": lo["formula"], "discharged": hi["formula"],
                "id_charged": lo["id"], "id_discharged": hi["id"],
                "x_range": f"{prof['x_lo']:.2f}-{prof['x_hi']:.2f}",
                "avg_V": round(prof["avg_V"], 3),
                "min_step_V": round(min(s["V"] for s in prof["usable"]), 3),
                "max_step_V": round(max(s["V"] for s in prof["usable"]), 3),
                "n_steps": len(prof["usable"]),
                "capacity_mAh_g": round(q_cap, 1),
                "dV_percent": round(dv, 1),
                "E_cell_Wh_kg": round(e_cell, 1),
                "E_vs_HC": round(e_cell / HC_ENERGY, 3),
                "all_converged": all(q["converged"] for q in p),
                "dft_avg_V": g.get("dft_avg_V"),
                "dft_capacity": g.get("dft_capacity"),
                "li_dft_avg_V": g.get("li_dft_avg_V"),
                "steps": json.dumps([{k: (round(v, 3) if isinstance(v, float) else v)
                                      for k, v in s.items()} for s in prof["steps"]]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("계산된 후보가 없습니다.")
    # 검증 단계에서 찾은 계통 오차: MP 가 +U 를 쓰는 전이금속의 산화물/불화물은
    # MatPES-PBE(무 U) M3GNet 이 전압을 0.8~0.9 V 낮게 예측 -> '신뢰도' 조건으로 제외
    def u_sensitive(formula) -> bool:
        if not isinstance(formula, str):
            return False
        els = {e.symbol for e in Composition(formula).elements}
        return bool(els & config.U_ELEMENTS) and bool(els & {"O", "F"})

    df["u_sensitive"] = df["discharged"].map(u_sensitive)
    conds = [  # (이름, 조건) — 순서대로 하나씩 더한다 (깔때기)
        (f"전압 {config.V_MIN}~{config.V_MAX} V", df["avg_V"].between(config.V_MIN, config.V_MAX)),
        ("Na 석출 없음 (최저 계단 > 0 V)", df["min_step_V"] > config.MIN_STEP_V),
        (f"용량 ≥ {config.MIN_CAPACITY:.0f} mAh/g", df["capacity_mAh_g"] >= config.MIN_CAPACITY),
        (f"부피 변화 ≤ {config.MAX_DV_PERCENT:.0f} %", df["dV_percent"].abs() <= config.MAX_DV_PERCENT),
        ("신뢰도 (+U 산화물 제외)", ~df["u_sensitive"]),
    ]
    keep = pd.Series(True, index=df.index)
    funnel = {"N0 계산한 전극계": int(len(df))}
    for i, (name, c) in enumerate(conds, 1):
        keep &= c.fillna(False)
        funnel[f"N{i} {name}"] = int(keep.sum())
    df["pass_stage2"] = keep
    df = df.sort_values(["pass_stage2", "E_cell_Wh_kg"], ascending=[False, False])
    df.to_csv(DATA / "stage2.csv", index=False)
    (DATA / "funnel_stage2.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=1))
    print("\n깔때기:", funnel)

    print(f"\n하드카본 기준 셀 에너지밀도(대리지표): {HC_ENERGY:.1f} Wh/kg (vs {config.CATHODE['name']})")
    cols = ["id", "kind", "charged", "discharged", "avg_V", "min_step_V", "capacity_mAh_g",
            "dV_percent", "E_cell_Wh_kg", "E_vs_HC", "dft_avg_V", "pass_stage2"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[[c for c in cols if c in df]].head(30).to_string(index=False))
    if df["dft_avg_V"].notna().any():
        d = df.dropna(subset=["dft_avg_V", "avg_V"])
        mae = (d["avg_V"] - d["dft_avg_V"]).abs().mean()
        print(f"\nM3GNet vs MP-DFT 평균전압 MAE = {mae:.3f} V  (n={len(d)})")
    print(f"\n저장: {DATA / 'stage2.csv'}  (통과 {int(df['pass_stage2'].sum())}개)")


if __name__ == "__main__":
    main()
