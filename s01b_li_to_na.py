"""1-b단계: Li → Na 치환 트랙 (가상 Na 음극 설계) -> data/candidates.json 에 'substituted' 추가

MP 의 Na 전극 DB 는 작다(평균전압 1.5 V 미만 21개). 그래서 Li 전지용 저전압 음극(Li 삽입 전극 DB)을
가져와 Li 를 Na 로 바꾼 '가상 Na 전극'을 만든다. 2단계에서 M3GNet 으로 격자째 다시 최적화하므로
Na 가 커서 늘어나는 부피도 자동으로 반영된다.

사용법:  python s01b_li_to_na.py     (s01 다음에 실행, MP_API_KEY 필요)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pymatgen.core import Structure

import config
from common import allowed

DATA = Path(__file__).parent / "data"
LI_MAX_VOLTAGE = 2.0  # V vs Li/Li+, Na 로 바꾸면 보통 전압이 낮아지므로 넓게 잡는다


def main():
    from mp_api.client import MPRester

    key = os.environ["MP_API_KEY"]
    data = json.loads((DATA / "candidates.json").read_text())
    data["substituted"] = []
    need: set[str] = set()
    n_raw = n_el = n_stab = 0

    with MPRester(key) as mpr:
        docs = mpr.materials.insertion_electrodes.search(
            working_ion="Li", average_voltage=(0.0, LI_MAX_VOLTAGE),
            fields=["framework_formula", "average_voltage", "capacity_grav", "max_delta_volume",
                    "stability_charge", "stability_discharge", "id_charge", "id_discharge",
                    "material_ids", "elements"],
        )
        n_raw = len(docs)
        for d in docs:
            els = [str(e) for e in d.elements if str(e) != "Li"]
            if "Na" in els or not allowed(els):
                continue
            n_el += 1
            if max(d.stability_charge or 0, d.stability_discharge or 0) > config.MP_MAX_EHULL:
                continue
            n_stab += 1
            ids = sorted({str(d.id_charge), str(d.id_discharge)} | {str(m) for m in d.material_ids or []})
            data["substituted"].append({
                "battery_id": f"sub_{d.framework_formula}_{d.id_charge}_{d.id_discharge}",
                "framework": str(d.framework_formula),
                "ids": [f"sub-{i}" for i in ids],
                "li_dft_avg_V": d.average_voltage,
                "li_dft_capacity": d.capacity_grav,
                "li_dft_dV": d.max_delta_volume,
            })
            need |= set(ids)

        got = {}
        need_l = sorted(need)
        for i in range(0, len(need_l), 500):
            for s in mpr.materials.summary.search(material_ids=need_l[i:i + 500],
                                                  fields=["material_id", "structure"]):
                got[str(s.material_id)] = s.structure

    n_struct = 0
    for mid, st in got.items():
        if len(st) > config.MAX_SITES:
            continue
        st = st.copy()
        if "Li" in [e.symbol for e in st.composition.elements]:
            frac_li = st.composition.get_atomic_fraction("Li")
            st.replace_species({"Li": "Na"})
            # Na(1.02 Å) > Li(0.76 Å): 초기 부피를 약간 키워 최적화가 쉽게 출발하도록 함
            st.scale_lattice(st.volume * (1.0 + 0.6 * frac_li))
        data["structures"][f"sub-{mid}"] = {"formula": st.composition.reduced_formula,
                                            "e_hull_dft": None, "structure": st.as_dict()}
        n_struct += 1

    data["substituted"] = [e for e in data["substituted"]
                           if all(i in data["structures"] for i in e["ids"])]
    data["funnel_substituted"] = {"Li 전극 (V<2.0 vs Li)": n_raw, "원소 제한": n_el,
                                  "안정성 (E_hull≤0.1)": n_stab,
                                  "원자 수 ≤ 60": len(data["substituted"])}
    (DATA / "candidates.json").write_text(json.dumps(data))
    print(data["funnel_substituted"], f"치환 구조 {n_struct}개")


if __name__ == "__main__":
    main()
