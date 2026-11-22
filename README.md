# 고전압 ESS용 소듐이온전지 음극재 스크리닝 (M3GNet + ASE)

하드카본을 대신할 Na 이온 음극재를 **범용 머신러닝 포텐셜(M3GNet)과 분자동역학(ASE)만으로** 찾는 파이프라인입니다.
소재 설계 수업 개인 프로젝트 결과물입니다.

## 질문

고전압 양극(NVPF, 3.9 V)을 쓰는 ESS 셀에서, 하드카본(약 0.25 V, 300 mAh/g)을 대신할 음극이 있는가?

| 조건 | 기준 | 이유 |
|---|---|---|
| 평균 전압 | 0.05 ~ 1.0 V vs Na/Na⁺ | 0 V 근처는 Na 석출 위험, 1 V 이상은 셀 전압 손실 |
| 전압 계단 | 모두 > 0 V | 충전 중 Na 금속 석출 방지 |
| 용량 | ≥ 150 mAh/g | 실용 용량 하한 |
| 부피 변화 | ≤ 30 % | ESS 장수명 (직접 정한 기준) |
| 신뢰도 | +U 원소 산화물/불화물 제외 | 검증에서 M3GNet 전압이 0.85 V 낮게 나온 계열 |

## 결과 요약

- 후보 133개 전극계 (MP Na 전극 19 + Na 합금계 10 + **Li 전극의 Li→Na 치환 104**), 구조 476개를 M3GNet으로 최적화했습니다.
- 깔때기: 133 → 50 (전압) → 50 (석출 없음) → 16 (용량) → 5 (부피) → **1 (신뢰도)**
- 최종 후보: **아나타제 TiO₂ → NaTi₂O₄**. 평균 0.89 V(Ti 산화물 오차를 보정하면 약 0.6 V), 168 mAh/g, 부피 변화 +20 %, 셀 에너지밀도는 하드카본의 67 %입니다.
- Sn, Sb, P 합금은 에너지밀도가 하드카본의 1.1~1.3배지만, 부피 변화가 +260~404 %입니다.
- MD (Langevin NVT): NaTi₂O₄의 Na 확산 활성화 에너지는 0.49 eV(1000~1300 K, 100 ps)이고, 300 K 확산계수는 약 1×10⁻¹² cm²/s입니다. 1시간 확산 길이는 약 0.7 µm이므로 입자를 작게 만들어야 합니다. Na₃Sb는 0.25 eV(약 2×10⁻⁸ cm²/s)로 약 2만 배 빠르고, Na₁₅Sn₄는 800 K에서 골격이 무너졌습니다.
- M3GNet과 MP-DFT 평균 전압의 MAE는 0.34 V입니다(n = 19). 황화물·합금은 정확하고, Ti 산화물은 약 +0.3 V, Mn 산화물은 약 −0.85 V 벗어납니다.

## 파이프라인

```
s01_fetch_candidates.py  MP: Na 삽입 전극 + Na–M 이원계
s01b_li_to_na.py         MP: Li 저전압 전극 → Li를 Na로 치환한 가상 구조 (설계 단계)
s02_relax_voltage.py     ASE FIRE + FrechetCellFilter 최적화 → 볼록 껍질 전압, 용량, ΔV → 깔때기
s03_md_diffusion.py      Langevin NVT MD (600/800/1000 K) → MSD → 확산계수, 아레니우스 활성화 에너지
s03b_tio2_long.py        최종 후보만 추가 MD (1000/1150/1300 K, 100 ps, Na 공공 2개)
s04_rank.py              종합 점수와 그림
s05_export.py            발표용 숫자 정리 (data/deck_data.json), 최종 후보 CIF
config.py                모든 기준값    common.py  전압·용량 계산    mlip.py  M3GNet 계산기
```

전압 = −[E(NaₓF) − E(F) − x·E(Na)] / x (F: Na를 뺀 골격, E(Na): bcc Na 금속)

## 환경

```bash
conda create -y -n naib python=3.12 && conda activate naib
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install matgl ase mp-api pandas matplotlib
```

확인한 버전: torch 2.14.1, matgl 4.1.4 (`M3GNet-PES-MatPES-PBE-2025.2`), ASE 3.29, pymatgen 2026.10, mp-api 0.47
(WSL2 Ubuntu, RTX 4060 Laptop GPU)

- matgl v3부터 DGL 백엔드가 없어지고 M3GNet이 PyG로 다시 구현되었습니다. 이전 모델 `M3GNet-MP-2021.2.8-PES`는 받을 수 없습니다.
- 최신 `PESCalculator`는 stress를 기본적으로 GPa로 반환하므로, ASE 셀 최적화에는 `stress_unit="eV/A3"`가 필요합니다(`mlip.py`에 반영).

## 실행

```bash
export MP_API_KEY=...        # 키는 코드나 파일에 넣지 않는다
python s01_fetch_candidates.py
python s01b_li_to_na.py
python s02_relax_voltage.py  # 약 20분 (476 구조)
python s03_md_diffusion.py --ids sub_TiO2_mp-390_mp-38280 Na-Sb Na-Sn Na-P   # 약 3시간
python s03b_tio2_long.py     # 약 2시간
python s04_rank.py
python s05_export.py
```

중간에 끊겨도 다시 실행하면 이어서 진행합니다(`data/relaxed/`, `data/md/`에 저장된 결과는 건너뜀).
모델이나 설정을 바꿨다면 두 폴더를 지우고 다시 실행하세요.

## 결과 파일 (`data/`)

| 파일 | 내용 |
|---|---|
| `stage2.csv` | 전극계별 전압, 용량, 부피 변화, 에너지밀도, 통과 여부 |
| `funnel_stage2.json` | 깔때기 단계별 후보 수 |
| `stage3.csv`, `md/` | MD 확산계수, 활성화 에너지, MSD 곡선 |
| `stage3_tio2_long.csv`, `md_long/` | 최종 후보 추가 MD |
| `ranking.csv` | 종합 점수 |
| `final_*.cif` | 최종 후보 구조 (충전/방전) |
| `relaxed/` | 모든 최적화 구조와 에너지 |

## 한계

- 열역학 전압만 계산합니다. 과전압, 히스테리시스, 초기 쿨롱 효율, SEI는 다루지 않습니다.
- 결정상만 다루므로 비정질인 하드카본 자체는 계산하지 못합니다.
- Li→Na 치환 구조는 가상 구조이고, 합성 가능성은 확인하지 않았습니다.
- 확산계수는 600~1300 K MD를 300 K로 외삽한 값입니다. NaTi₂O₄는 1000 K와 1150 K 값이 비슷해 활성화 에너지의 오차가 큽니다.
- 기본 MD(40 ps, 600~1000 K)에서는 NaTi₂O₄의 Na가 2 Å² 미만으로만 움직여, 활성화 에너지를 추가 MD로만 구했습니다.

데이터 출처: Materials Project (CC BY 4.0)
