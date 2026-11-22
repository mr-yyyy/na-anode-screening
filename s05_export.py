"""5단계: 발표용 숫자와 그림 데이터를 한 파일로 정리 -> data/deck_data.json, data/fig_*.png

사용법:  python s05_export.py     (s02, s03 이후)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure

import config
from common import HC_ENERGY

DATA = Path(__file__).parent / "data"
FINAL_ID = "sub_TiO2_mp-390_mp-38280"
KB = 8.617333262e-5


def category(formula: str) -> str:
    els = {e.symbol for e in Composition(formula).elements}
    if els & config.U_ELEMENTS and els & {"O", "F"}:
        return "+U 산화물"
    if "Ti" in els and "O" in els:
        return "Ti 산화물"
    return "그 외 (황화물 등)"


def pretty(formula: str) -> str:
    return Composition(formula).reduced_formula


def main():
    cand = json.loads((DATA / "candidates.json").read_text())
    s2 = pd.read_csv(DATA / "stage2.csv")
    funnel = json.loads((DATA / "funnel_stage2.json").read_text())
    out: dict = {"hc_energy": HC_ENERGY, "cathode": config.CATHODE, "hard_carbon": config.HARD_CARBON}

    out["sources"] = {
        "na_electrodes": cand["funnel_na"],
        "alloy_systems": len(cand["alloy"]),
        "li_to_na": cand["funnel_substituted"],
        "n_structures": len(cand["structures"]) + 1,
    }
    out["funnel"] = [{"label": k.split(" ", 1)[1], "n": v, "step": k.split(" ", 1)[0]}
                     for k, v in funnel.items()]

    # ---- 검증: M3GNet vs MP-DFT ----
    v = s2.dropna(subset=["dft_avg_V", "avg_V"]).copy()
    v["err"] = v["avg_V"] - v["dft_avg_V"]
    v["cat"] = v["discharged"].map(category)
    out["parity"] = [{"x": round(r.dft_avg_V, 3), "y": round(r.avg_V, 3), "cat": r.cat,
                      "label": f"{pretty(r.charged)}→{pretty(r.discharged)}"} for r in v.itertuples()]
    out["parity_mae"] = round(float(v["err"].abs().mean()), 3)
    out["parity_by_cat"] = {c: {"n": int(len(g)), "mean_err": round(float(g["err"].mean()), 2)}
                            for c, g in v.groupby("cat")}

    # ---- 결과: 에너지밀도 vs 부피 변화 ----
    t = s2.dropna(subset=["E_vs_HC", "dV_percent"]).copy()
    t = t[t["avg_V"].between(0.0, 1.0)]  # 음극으로 쓸 수 있는 전압 범위만 표시
    # 충전 상태가 단일 원소(Sn, Sb, ...)이면 합금 반응으로 분류
    is_alloy = (t["kind"] == "alloy") | t["charged"].map(lambda f: len(Composition(f).elements) == 1)
    t["group"] = np.where(t["id"] == FINAL_ID, "최종 후보", np.where(is_alloy, "합금", "삽입형"))
    t = t.sort_values("E_vs_HC", ascending=False).drop_duplicates(subset=["discharged"])
    out["tradeoff"] = [{"dV": round(abs(r.dV_percent), 1), "E": round(r.E_vs_HC, 3), "group": r.group,
                        "label": pretty(r.discharged), "V": r.avg_V, "Q": r.capacity_mAh_g,
                        "u": bool(r.u_sensitive)} for r in t.itertuples()]

    # ---- 최종 후보 ----
    f = s2.set_index("id").loc[FINAL_ID]
    out["final"] = {k: (None if isinstance(f[k], float) and math.isnan(f[k]) else f[k]) for k in
                    ["charged", "discharged", "id_charged", "id_discharged", "avg_V", "capacity_mAh_g",
                     "dV_percent", "E_cell_Wh_kg", "E_vs_HC", "li_dft_avg_V", "steps"]}
    ti = v[v["cat"] == "Ti 산화물"]["err"]
    out["ti_oxide_bias"] = round(float(ti.median()), 2)

    # ---- 합금 비교 ----
    out["alloys"] = []
    for cid in ["Na-Sn", "Na-Sb", "Na-P"]:
        r = s2.set_index("id").loc[cid]
        out["alloys"].append({"id": cid, "discharged": pretty(r.discharged), "V": r.avg_V,
                              "Q": r.capacity_mAh_g, "dV": r.dV_percent, "E": r.E_vs_HC})

    # ---- MD ----
    # 기본 MD(data/md) + 최종 후보 추가 MD(data/md_long, s03b). 최종 후보는 추가 MD 값을 쓴다.
    long_csv = DATA / "stage3_tio2_long.csv"
    has_long = long_csv.exists()
    md = []
    for folder, src in (("md", "base"), ("md_long", "long")):
        for p in sorted((DATA / folder).glob("*K.json")) if (DATA / folder).exists() else []:
            name, T = p.stem.rsplit("_", 1)
            if has_long and name == FINAL_ID and src == "base":
                continue
            md.append({"id": name, "T": int(T[:-1]), "src": src} | json.loads(p.read_text()))
    s3 = pd.read_csv(DATA / "stage3.csv") if (DATA / "stage3.csv").exists() else pd.DataFrame()
    if has_long:
        s3 = pd.concat([s3[s3["id"] != FINAL_ID] if len(s3) else s3, pd.read_csv(long_csv)],
                       ignore_index=True)
    out["md"] = md
    out["md_summary"] = s3.replace({np.nan: None}).to_dict("records")

    (DATA / "deck_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ["sources", "funnel", "parity_mae", "parity_by_cat",
                                         "final", "ti_oxide_bias", "alloys"]},
                     ensure_ascii=False, indent=1, default=str))
    print("MD summary:", out["md_summary"])

    # 최종 후보 구조를 CIF 로 저장 (공개 저장소용)
    for key in ("id_charged", "id_discharged"):
        s = Structure.from_dict(json.loads((DATA / "relaxed" / f"{f[key]}.json").read_text())["structure"])
        s.to(filename=str(DATA / f"final_{s.composition.reduced_formula}_{f[key]}.cif"))
    print("최종 후보 CIF 저장")


if __name__ == "__main__":
    main()
