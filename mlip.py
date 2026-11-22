"""M3GNet ASE 계산기 생성 (matgl PESCalculator)."""

from __future__ import annotations

import matgl
import torch
from matgl.ext.ase import PESCalculator

import config


def get_calculator() -> PESCalculator:
    device = config.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    pot = matgl.load_model(config.M3GNET_MODEL).to(device)
    print(f"M3GNet: {config.M3GNET_MODEL} on {device}")
    # 최신 matgl 은 stress 기본 단위가 GPa -> ASE 셀 최적화에는 반드시 eV/Å^3 로 설정
    return PESCalculator(pot, stress_unit="eV/A3")
