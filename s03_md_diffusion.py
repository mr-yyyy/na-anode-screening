"""3단계: ASE NVT-Langevin MD (M3GNet) -> Na 확산계수, 활성화에너지, 골격 안정성 -> data/stage3.csv

- 2단계 통과 후보 중 상위 N개의 '완전 소듐화 상'으로 슈퍼셀을 만든다 (Na 공공 1개 도입)
- (후보 × 온도) 조합을 하나씩 실행. 끝난 조합은 data/md/*.json 에 저장되어 재실행 시 건너뜀
- 최소 이미지 규약으로 변위를 누적해 unwrapped 궤적을 만들고 다중 시간원점 MSD 계산
- D(T) 아레니우스 피팅 -> Ea, D(300 K)
- 골격 원자 MSD 가 크면 그 온도에서 구조 붕괴/용융으로 판정 (해당 온도 D는 피팅에서 제외)

사용법:
    python s03_md_diffusion.py                 # stage2 통과 상위 MD_TOP_N 개
    python s03_md_diffusion.py --ids ID1 ID2   # 특정 후보만
    python s03_md_diffusion.py --include-failed  # stage2 탈락 후보도 상위 N개에 포함 (비교용)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import time

import numpy as np
import pandas as pd
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import config
from mlip import get_calculator

DATA = Path(__file__).parent / "data"
MD_DIR = DATA / "md"
KB_EV = 8.617333262e-5
A2PS_TO_CM2S = 1e-4


def make_supercell(s: Structure, seed: int = 0) -> Structure:
    lat = s.lattice.matrix
    vol = abs(np.linalg.det(lat))
    heights = [vol / np.linalg.norm(np.cross(lat[(i + 1) % 3], lat[(i + 2) % 3])) for i in range(3)]
    n = [max(1, math.ceil(config.MD_MIN_LENGTH / h)) for h in heights]
    while np.prod(n) * len(s) > config.MD_MAX_ATOMS and max(n) > 1:
        n[int(np.argmax(n))] -= 1  # 원자 수 상한 우선 (이 경우 cutoff 대비 셀이 작다는 점에 유의)
    sc = s.copy()
    sc.make_supercell(n)
    na_idx = [i for i, site in enumerate(sc) if site.specie.symbol == "Na"]
    rng = np.random.default_rng(seed)
    k = min(config.MD_NA_VACANCIES, max(0, len(na_idx) - 1))
    if k:
        sc.remove_sites(sorted(rng.choice(na_idx, size=k, replace=False).tolist()))
    return sc


def msd_curve(traj: np.ndarray, max_frac: float = 0.5) -> np.ndarray:
    """traj: (n_frames, n_atoms, 3) unwrapped. 다중 시간원점 평균 MSD (lag = 0..max_lag)."""
    nf = traj.shape[0]
    max_lag = max(2, int(nf * max_frac))
    out = np.zeros(max_lag)
    for lag in range(1, max_lag):
        d = traj[lag:] - traj[:-lag]
        out[lag] = (d ** 2).sum(-1).mean()
    return out


def fit_D(msd: np.ndarray, dt_frame_ps: float) -> tuple[float, float]:
    """MSD 의 20~100% lag 구간 선형 피팅 -> D [cm^2/s], R^2"""
    lags = np.arange(len(msd)) * dt_frame_ps
    i0 = max(1, int(0.2 * len(msd)))
    x, y = lags[i0:], msd[i0:]
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return max(slope, 0.0) / 6.0 * A2PS_TO_CM2S, float(r2)


def run_md(job: dict, calc) -> None:
    """job: {"name", "T", "structure"} 하나를 NVT MD. 결과를 job 에 기록하고 파일로 저장."""
    MD_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{job['name']}_{job['T']}K"
    res_file = MD_DIR / f"{tag}.json"
    if res_file.exists():
        job.update(json.loads(res_file.read_text()))
        print(f"  {tag}: 기존 결과 사용")
        return

    atoms = AseAtomsAdaptor.get_atoms(job["structure"])
    atoms.calc = calc
    rng = np.random.default_rng(42)
    MaxwellBoltzmannDistribution(atoms, temperature_K=job["T"], rng=rng)
    Stationary(atoms)
    dyn = Langevin(atoms, timestep=config.MD_TIMESTEP_PS * 1000 * units.fs,
                   temperature_K=job["T"], friction=config.MD_FRICTION_PER_FS / units.fs,
                   rng=rng, logfile=None)

    t0 = time.time()
    dyn.run(config.MD_EQUIL_STEPS)
    cell = np.array(atoms.get_cell())
    inv = np.linalg.inv(cell)
    prev = atoms.get_positions()
    unwrapped = prev.copy()
    frames = [unwrapped.copy()]
    n_chunks = config.MD_PROD_STEPS // config.MD_SAMPLE_EVERY
    for c in range(1, n_chunks + 1):
        dyn.run(config.MD_SAMPLE_EVERY)
        cur = atoms.get_positions()
        frac = (cur - prev) @ inv
        frac -= np.round(frac)  # 최소 이미지: 경계를 넘어 다시 들어온 원자 보정
        unwrapped = unwrapped + frac @ cell
        prev = cur
        frames.append(unwrapped.copy())
        if c == 50:
            eta = (time.time() - t0) / (config.MD_EQUIL_STEPS + 50 * config.MD_SAMPLE_EVERY) * (
                config.MD_PROD_STEPS - 50 * config.MD_SAMPLE_EVERY) / 60
            print(f"    {tag}: 남은 시간 약 {eta:.0f}분")

    traj = np.stack(frames)  # (n_frames, n_atoms, 3)
    nums = atoms.get_atomic_numbers()
    dt_frame = config.MD_TIMESTEP_PS * config.MD_SAMPLE_EVERY
    msd_na = msd_curve(traj[:, nums == 11])
    msd_fw = msd_curve(traj[:, nums != 11])
    D, r2 = fit_D(msd_na, dt_frame)
    res = {
        "D_cm2_s": D, "R2": r2,
        "msd_na_final": float(msd_na[-1]), "msd_fw_final": float(msd_fw[-1]),
        "framework_stable": bool(msd_fw[-1] < config.FRAMEWORK_MSD_LIMIT),
        "na_diffusive": bool(msd_na[-1] >= config.NA_MSD_MIN),
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    job.update(res)
    res_file.write_text(json.dumps(res))
    pd.DataFrame({"t_ps": np.arange(len(msd_na)) * dt_frame, "msd_na": msd_na,
                  "msd_framework": msd_fw}).to_csv(MD_DIR / f"{tag}_msd.csv", index=False)
    print(f"  {tag}: D={D:.2e} cm2/s, MSD_Na={msd_na[-1]:.2f}, MSD_fw={msd_fw[-1]:.2f} Å², "
          f"{res['wall_min']} min")


def arrhenius(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["framework_stable"] and r["na_diffusive"] and r["D_cm2_s"] > 0]
    res = {"n_T_used": len(ok)}
    if len(ok) < 2:
        return res | {"Ea_eV": math.nan, "D300_cm2_s": math.nan, "log10_D300": math.nan}
    invT = np.array([1.0 / r["T"] for r in ok])
    lnD = np.log([r["D_cm2_s"] for r in ok])
    slope, icpt = np.polyfit(invT, lnD, 1)
    ea = -slope * KB_EV
    d300 = math.exp(icpt + slope / 300.0)
    return res | {"Ea_eV": round(float(ea), 3), "D300_cm2_s": d300,
                  "log10_D300": round(math.log10(d300), 2) if d300 > 0 else math.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--include-failed", action="store_true")
    args = ap.parse_args()

    st2 = pd.read_csv(DATA / "stage2.csv")
    st2 = st2.dropna(subset=["id_discharged"])
    if args.ids:
        sel = st2[st2["id"].isin(args.ids)]
    else:
        pool = st2 if args.include_failed else st2[st2["pass_stage2"]]
        sel = pool.sort_values("E_cell_Wh_kg", ascending=False).head(config.MD_TOP_N)
    if sel.empty:
        raise SystemExit("MD 대상 후보가 없습니다. (--include-failed 또는 config 기준 완화)")

    jobs = []
    for _, r in sel.iterrows():
        d = json.loads((DATA / "relaxed" / f"{r['id_discharged']}.json").read_text())
        sc = make_supercell(Structure.from_dict(d["structure"]))
        if not any(site.specie.symbol == "Na" for site in sc):
            continue
        for T in config.MD_TEMPS:
            jobs.append({"name": r["id"], "T": T, "structure": sc, "n_atoms": len(sc),
                         "formula": sc.composition.reduced_formula})

    calc = get_calculator()
    total_ps = config.MD_PROD_STEPS * config.MD_TIMESTEP_PS
    print(f"MD: 후보 {len(sel)}개 × 온도 {config.MD_TEMPS} -> 실행 {len(jobs)}개, "
          f"production {total_ps:.0f} ps")
    for i, j in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {j['name']} ({j['formula']}, {j['n_atoms']} atoms) @ {j['T']} K")
        run_md(j, calc)

    out = []
    for name in dict.fromkeys(j["name"] for j in jobs):
        rows = [j for j in jobs if j["name"] == name]
        rec = {"id": name, "md_formula": rows[0]["formula"], "md_atoms": rows[0]["n_atoms"]}
        for r in rows:
            rec[f"D_{r['T']}K"] = r["D_cm2_s"]
            rec[f"fw_stable_{r['T']}K"] = r["framework_stable"]
            rec[f"msdNa_{r['T']}K"] = round(r["msd_na_final"], 2)
        rec["framework_stable_all"] = all(r["framework_stable"] for r in rows)
        rec |= arrhenius(rows)
        out.append(rec)
    df = pd.DataFrame(out).sort_values("log10_D300", ascending=False)
    df.to_csv(DATA / "stage3.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[["id", "md_formula", "md_atoms", "Ea_eV", "log10_D300", "n_T_used",
                  "framework_stable_all"]].to_string(index=False))
    print(f"\n저장: {DATA / 'stage3.csv'}")


if __name__ == "__main__":
    main()
