"""4단계: 2·3단계 결과 종합 -> 점수/순위 -> data/ranking.csv, data/*.png

점수 (0~1, 가중합; 가중치는 아래 W 에서 조정):
  energy    : 셀 에너지밀도 대리지표 / 하드카본 값   (고전압 셀에서 낮은 음극 전위 + 용량)
  volume    : 1 - |ΔV|/MAX_DV_PERCENT                (ESS 장수명)
  diffusion : log10 D(300K) 를 [-14, -8] 구간에서 0~1 로 정규화  (출력 특성)
  stability : 모든 MD 온도에서 골격이 유지되면 1, 아니면 0.5
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from common import HC_ENERGY, energy_density_proxy  # noqa: E402

DATA = Path(__file__).parent / "data"
W = {"energy": 0.45, "volume": 0.20, "diffusion": 0.25, "stability": 0.10}


def main():
    s2 = pd.read_csv(DATA / "stage2.csv")
    p3 = DATA / "stage3.csv"
    s3 = pd.read_csv(p3) if p3.exists() else pd.DataFrame({"id": []})
    p3b = DATA / "stage3_tio2_long.csv"  # 최종 후보는 추가 MD(s03b) 값으로 대체
    if p3b.exists():
        long = pd.read_csv(p3b)
        long["framework_stable_all"] = long.filter(like="fw_stable_").all(axis=1)
        s3 = pd.concat([s3[~s3["id"].isin(long["id"])], long], ignore_index=True)
    df = s2.merge(s3, on="id", how="left")
    for c in ("log10_D300", "framework_stable_all"):
        if c not in df:
            df[c] = np.nan

    e = (df["E_cell_Wh_kg"] / HC_ENERGY).clip(0, 1.5) / 1.5
    v = (1 - df["dV_percent"].abs() / config.MAX_DV_PERCENT).clip(0, 1)
    d = ((df["log10_D300"] + 14) / 6).clip(0, 1)
    st = df["framework_stable_all"].map({True: 1.0, False: 0.5})
    df["score"] = (W["energy"] * e + W["volume"] * v
                   + W["diffusion"] * d.fillna(0) + W["stability"] * st.fillna(0))
    df["md_done"] = df["log10_D300"].notna()
    df = df.sort_values(["pass_stage2", "md_done", "score"], ascending=False)
    df.to_csv(DATA / "ranking.csv", index=False)

    cols = ["id", "kind", "charged", "discharged", "avg_V", "capacity_mAh_g", "dV_percent",
            "E_vs_HC", "Ea_eV", "log10_D300", "framework_stable_all", "pass_stage2", "score"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df[[c for c in cols if c in df]].head(20).to_string(index=False))

    # --- 그림 1: 전압-용량 맵 + 등에너지밀도 곡선 ---
    fig, ax = plt.subplots(figsize=(7, 5))
    q = np.linspace(50, 1000, 300)
    for ratio in (0.8, 1.0, 1.2):
        # (Vc - Va) * Qc*Qa/(Qc+Qa) = ratio * HC  ->  Va
        va = config.CATHODE["voltage"] - ratio * HC_ENERGY * (config.CATHODE["capacity"] + q) / (
            config.CATHODE["capacity"] * q)
        ax.plot(q, va, ls="--", lw=0.8, color="gray")
        ax.text(q[-1], va[-1], f" {ratio:.1f}×HC", fontsize=8, color="gray", va="center")
    m = df["pass_stage2"]
    ax.scatter(df.loc[~m, "capacity_mAh_g"], df.loc[~m, "avg_V"], s=18, c="lightgray",
               label="stage2 fail")
    sc = ax.scatter(df.loc[m, "capacity_mAh_g"], df.loc[m, "avg_V"], s=36,
                    c=df.loc[m, "dV_percent"].abs(), cmap="viridis", label="stage2 pass")
    plt.colorbar(sc, ax=ax, label="|ΔV| (%)")
    ax.scatter([config.HARD_CARBON["capacity"]], [config.HARD_CARBON["voltage"]], marker="*",
               s=220, c="red", label="Hard carbon (exp.)")
    for _, r in df[m].head(10).iterrows():
        ax.annotate(r["discharged"], (r["capacity_mAh_g"], r["avg_V"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhspan(config.V_MIN, config.V_MAX, color="tab:blue", alpha=0.06)
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.2, 2.0)
    ax.set_xlabel("Capacity (mAh/g, desodiated host)")
    ax.set_ylabel("Average voltage vs Na/Na$^+$ (V, M3GNet)")
    ax.set_title(f"Anode screening vs {config.CATHODE['name']} cathode")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(DATA / "voltage_capacity_map.png", dpi=200)

    # --- 그림 2: M3GNet vs DFT 전압 패리티 ---
    dd = df.dropna(subset=["dft_avg_V", "avg_V"])
    if len(dd):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(dd["dft_avg_V"], dd["avg_V"], s=15)
        lim = [min(dd["dft_avg_V"].min(), dd["avg_V"].min()) - 0.2,
               max(dd["dft_avg_V"].max(), dd["avg_V"].max()) + 0.2]
        ax.plot(lim, lim, "k--", lw=0.8)
        mae = (dd["avg_V"] - dd["dft_avg_V"]).abs().mean()
        ax.set_title(f"M3GNet vs MP-DFT (MAE {mae:.2f} V, n={len(dd)})")
        ax.set_xlabel("DFT average voltage (V)")
        ax.set_ylabel("M3GNet average voltage (V)")
        fig.tight_layout()
        fig.savefig(DATA / "parity_voltage.png", dpi=200)

    print(f"\n하드카본 대리지표 {HC_ENERGY:.1f} Wh/kg, "
          f"예) 0.5 V/250 mAh/g -> {energy_density_proxy(0.5, 250):.1f} Wh/kg")
    print(f"저장: {DATA / 'ranking.csv'}, voltage_capacity_map.png, parity_voltage.png")


if __name__ == "__main__":
    main()
