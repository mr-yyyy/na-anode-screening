"""스크리닝 기준과 계산 설정 (여기만 고치면 전체 파이프라인에 반영됨).

목표: 고전압 양극과 짝을 이루는 ESS용 소듐이온전지 음극재 (하드카본 제외)
  셀 전압 = V_cathode - V_anode  ->  음극 전위가 낮을수록 셀 에너지밀도가 올라감
  ESS          ->  긴 수명(작은 부피 변화), 저가 원소, 60 °C 근처의 열적 안정성
"""

# ----- MLIP -----
# matgl v3+ (PyG 백엔드) 의 M3GNet. 가중치는 Hugging Face 'materialyze' 에서 자동 다운로드.
# (구버전 M3GNet-MP-2021.2.8-PES 는 DGL 전용이고 더 이상 배포되지 않음)
M3GNET_MODEL = "M3GNet-PES-MatPES-PBE-2025.2"
DEVICE = None  # None 이면 cuda 를 자동 감지

# ----- 기준 양극 (에너지밀도 대리지표 계산용) -----
# Na3V2(PO4)2F3 (NVPF): 대표적인 고전압 폴리음이온 양극, ESS 후보
CATHODE = {"name": "NVPF", "voltage": 3.90, "capacity": 128.0}  # V vs Na/Na+, mAh/g

# ----- 하드카본 벤치마크 (실험값 기반 대략치) -----
HARD_CARBON = {"name": "Hard carbon", "voltage": 0.25, "capacity": 300.0, "dV_percent": 5.0}

# ----- 원소 제한 -----
# 비싸거나 희소한 원소, 독성/방사성 원소, Li/Co/Ni (Na 전지의 원가 장점을 없앰)
BANNED_ELEMENTS = {
    "Li", "Co", "Ni", "Be", "Cd", "Hg", "Pb", "Tl", "As",
    "Ag", "Au", "Pt", "Pd", "Rh", "Ru", "Ir", "Os", "Re",
    "Ga", "Ge", "In", "Te", "Sc", "Hf", "Ta",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Tc", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    "He", "Ne", "Ar", "Kr", "Xe",
}

# ----- 후보 수집 (01) -----
MP_MAX_DFT_VOLTAGE = 1.5  # MP 전극 DB에서 가져올 평균전압 상한 (넓게 잡고 M3GNet으로 다시 계산)
MP_MAX_EHULL = 0.10  # eV/atom, 준안정상까지 허용
MAX_SITES = 60  # 단위셀 원자 수 상한 (계산량 제한)
ALLOY_HOSTS = ["Sn", "Sb", "P", "Bi", "Si", "Al", "Zn", "Mg", "S", "Se"]  # Na-M 이원계 합금/화합물 트랙

# ----- 구조 최적화 (02) -----
RELAX_FMAX = 0.05  # eV/Å
RELAX_MAX_STEPS = 1500

# ----- 1차 필터 (02 결과에 적용) -----
V_MIN = 0.05  # V, 이보다 낮으면 Na 석출(덴드라이트) 위험, 특히 고전압 셀에서 급속충전 시
V_MAX = 1.00  # V, 이보다 높으면 셀 전압 손실이 커서 '고전압 셀'의 의미가 없어짐
MIN_STEP_V = 0.0  # 전압 계단 중 가장 낮은 값이 0 V 아래면 탈락
MIN_CAPACITY = 150.0  # mAh/g
MAX_DV_PERCENT = 30.0  # %, ESS 장수명 기준 (합금은 대부분 여기서 걸림 -> 값을 바꿔가며 trade-off 분석)
# MP 가 GGA+U 를 적용하는 원소. 이들의 산화물/불화물은 MatPES-PBE(무 U) M3GNet 전압이 계통적으로 낮음
U_ELEMENTS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Mo", "W"}

# ----- MD 확산 (03) -----
MD_TEMPS = [600, 800, 1000]  # K, 고온 MD -> 아레니우스 외삽으로 300 K 확산계수 추정
MD_TIMESTEP_PS = 0.002  # 2 fs
MD_EQUIL_STEPS = 2000  # 4 ps
MD_PROD_STEPS = 20000  # 40 ps (결과 품질을 높이려면 50000 이상 권장)
MD_SAMPLE_EVERY = 20  # 위치 저장 간격 (step)
MD_MIN_LENGTH = 10.0  # Å, 슈퍼셀 최소 길이 (M3GNet cutoff 5 Å의 2배)
MD_MAX_ATOMS = 160  # 슈퍼셀 원자 수 상한
MD_FRICTION_PER_FS = 0.01  # Langevin 마찰계수 (1/fs)
MD_NA_VACANCIES = 1  # 완전 소듐화상에서 Na 1개를 빼서 공공 매개 확산을 허용
MD_TOP_N = 10  # 02 단계 상위 몇 개를 MD로 넘길지 (ASE는 하나씩 순차 실행)
FRAMEWORK_MSD_LIMIT = 1.0  # Å^2, 골격 원자 MSD가 이보다 크면 해당 온도에서 구조 붕괴/용융으로 판정
NA_MSD_MIN = 2.0  # Å^2, Na MSD가 이보다 작으면 확산이 통계적으로 불충분 -> 아레니우스 피팅에서 제외
