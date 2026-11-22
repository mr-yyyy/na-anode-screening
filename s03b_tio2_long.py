"""3-b단계: 최종 후보(TiO2 -> NaTi2O4) 추가 MD -> data/stage3_tio2_long.csv

기본 MD(40 ps, 600~1000 K, Na 공공 1개)에서는 NaTi2O4 의 Na MSD 가 1 Å² 미만이라
활성화 에너지를 구할 수 없었다. 그래서 온도를 올리고(1000/1150/1300 K) 시간을 늘리고(100 ps)
Na 공공을 2개로 늘려(충방전 중간 조성 x≈0.44 에 해당) 다시 계산한다.

사용법:  python s03b_tio2_long.py
"""

from __future__ import annotations

import json

import pandas as pd
from pymatgen.core import Structure

import config
import s03_md_diffusion as md

FINAL_ID = "sub_TiO2_mp-390_mp-38280"
TEMPS = [1000, 1150, 1300]

config.MD_PROD_STEPS = 50000  # 100 ps
config.MD_NA_VACANCIES = 2
md.MD_DIR = md.DATA / "md_long"


def main():
    st2 = pd.read_csv(md.DATA / "stage2.csv").set_index("id")
    r = st2.loc[FINAL_ID]
    d = json.loads((md.DATA / "relaxed" / f"{r['id_discharged']}.json").read_text())
    sc = md.make_supercell(Structure.from_dict(d["structure"]))
    calc = md.get_calculator()
    jobs = [{"name": FINAL_ID, "T": T, "structure": sc, "n_atoms": len(sc),
             "formula": sc.composition.reduced_formula} for T in TEMPS]
    print(f"추가 MD: {sc.composition.reduced_formula} ({len(sc)} atoms), {TEMPS} K, "
          f"{config.MD_PROD_STEPS * config.MD_TIMESTEP_PS:.0f} ps")
    for i, j in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {j['T']} K")
        md.run_md(j, calc)
    rec = {"id": FINAL_ID, "md_formula": jobs[0]["formula"], "md_atoms": len(sc)}
    for j in jobs:
        rec[f"D_{j['T']}K"] = j["D_cm2_s"]
        rec[f"fw_stable_{j['T']}K"] = j["framework_stable"]
        rec[f"msdNa_{j['T']}K"] = round(j["msd_na_final"], 2)
    rec |= md.arrhenius(jobs)
    pd.DataFrame([rec]).to_csv(md.DATA / "stage3_tio2_long.csv", index=False)
    print(rec)


if __name__ == "__main__":
    main()
