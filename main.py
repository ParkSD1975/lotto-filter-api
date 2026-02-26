"""
lotto-filter-api / main.py (v3)
매트릭스 파일을 읽어서 필터 카운팅만 수행 - 메모리 최소화
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI(title="Lotto Filter Count API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

combos_mat  = None  # (8145060, 41) int8
combos_ac   = None  # (8145060,) int8
combos_nums = None  # (8145060, 6) int8

# 컬럼 인덱스
C_SUM=0; C_TAIL=1; C_ODD=2; C_HIGH=3
C_PRIME=4; C_COMP=5; C_SQ=6; C_TRI=7; C_TWIN=8; C_CONS=9
C_TDIG=10  # +0~9
C_BAND=20  # +0~4
C_PAL=25   # +0~8
C_ROW=34   # +0~6


@app.on_event("startup")
def startup_event():
    global combos_mat, combos_ac, combos_nums
    print("📂 매트릭스 파일 로딩 중...")
    base = os.path.dirname(os.path.abspath(__file__))
    combos_mat  = np.load(os.path.join(base, "combos_mat.npy"))
    combos_ac   = np.load(os.path.join(base, "combos_ac.npy"))
    combos_nums = np.load(os.path.join(base, "combos_nums.npy"))
    total_mb = (combos_mat.nbytes + combos_ac.nbytes + combos_nums.nbytes) / 1024 / 1024
    print(f"✅ 로딩 완료: {len(combos_mat):,}개 조합, {total_mb:.1f}MB")


class FilterRequest(BaseModel):
    fixed: list[int] = []
    excluded: list[int] = []
    total_sum_enabled: bool = False
    total_sum_min: int = 21
    total_sum_max: int = 255
    total_sum_excluded: list[int] = []
    last_digit_sum_enabled: bool = False
    last_digit_sum_min: int = 2
    last_digit_sum_max: int = 52
    last_digit_sum_excluded: list[int] = []
    ac_value_enabled: bool = False
    ac_value_min: int = 0
    ac_value_max: int = 10
    ac_value_excluded: list[int] = []
    odd_even_enabled: bool = False
    odd_even_counts: list[int] = []
    high_low_enabled: bool = False
    high_low_counts: list[int] = []
    prime_enabled: bool = False
    prime_counts: list[int] = []
    composite_enabled: bool = False
    composite_counts: list[int] = []
    square_enabled: bool = False
    square_counts: list[int] = []
    triangular_enabled: bool = False
    triangular_counts: list[int] = []
    twin_enabled: bool = False
    twin_counts: list[int] = []
    consecutive_enabled: bool = False
    consecutive_counts: list[int] = []
    tail_digit_enabled: bool = False
    tail_digit_filters: dict = {}
    band_enabled: bool = False
    band_filters: dict = {}
    palace_enabled: bool = False
    palace_filters: dict = {}
    paper_enabled: bool = False
    paper_filters: dict = {}
    regression_filters: list[dict] = []


@app.post("/api/count")
def count_combos(req: FilterRequest):
    if combos_mat is None:
        return {"count": 0, "error": "초기화 중입니다."}

    mat  = combos_mat
    ac   = combos_ac
    nums = combos_nums
    mask = np.ones(len(mat), dtype=bool)

    # 고정수/제외수
    for n in req.fixed:
        if 1 <= n <= 45:
            mask &= np.any(nums == np.int8(n), axis=1)
    for n in req.excluded:
        if 1 <= n <= 45:
            mask &= ~np.any(nums == np.int8(n), axis=1)

    # 총합 (오프셋 -128)
    if req.total_sum_enabled:
        smin = np.int8(req.total_sum_min - 128)
        smax = np.int8(req.total_sum_max - 128)
        m = (mat[:, C_SUM] >= smin) & (mat[:, C_SUM] <= smax)
        if req.total_sum_excluded:
            m &= ~np.isin(mat[:, C_SUM], np.array([v-128 for v in req.total_sum_excluded], dtype=np.int8))
        mask &= m

    # 끝수합
    if req.last_digit_sum_enabled:
        m = (mat[:, C_TAIL] >= req.last_digit_sum_min) & (mat[:, C_TAIL] <= req.last_digit_sum_max)
        if req.last_digit_sum_excluded:
            m &= ~np.isin(mat[:, C_TAIL], np.array(req.last_digit_sum_excluded, dtype=np.int8))
        mask &= m

    # AC값
    if req.ac_value_enabled:
        m = (ac >= req.ac_value_min) & (ac <= req.ac_value_max)
        if req.ac_value_excluded:
            m &= ~np.isin(ac, np.array(req.ac_value_excluded, dtype=np.int8))
        mask &= m

    if req.odd_even_enabled and req.odd_even_counts:
        mask &= np.isin(mat[:, C_ODD], np.array(req.odd_even_counts, dtype=np.int8))
    if req.high_low_enabled and req.high_low_counts:
        mask &= np.isin(mat[:, C_HIGH], np.array(req.high_low_counts, dtype=np.int8))
    if req.prime_enabled and req.prime_counts:
        mask &= np.isin(mat[:, C_PRIME], np.array(req.prime_counts, dtype=np.int8))
    if req.composite_enabled and req.composite_counts:
        mask &= np.isin(mat[:, C_COMP], np.array(req.composite_counts, dtype=np.int8))
    if req.square_enabled and req.square_counts:
        mask &= np.isin(mat[:, C_SQ], np.array(req.square_counts, dtype=np.int8))
    if req.triangular_enabled and req.triangular_counts:
        mask &= np.isin(mat[:, C_TRI], np.array(req.triangular_counts, dtype=np.int8))
    if req.twin_enabled and req.twin_counts:
        mask &= np.isin(mat[:, C_TWIN], np.array(req.twin_counts, dtype=np.int8))
    if req.consecutive_enabled and req.consecutive_counts:
        mask &= np.isin(mat[:, C_CONS], np.array(req.consecutive_counts, dtype=np.int8))

    if req.tail_digit_enabled and req.tail_digit_filters:
        for d_str, rng in req.tail_digit_filters.items():
            d = int(d_str)
            if 0 <= d <= 9:
                col = C_TDIG + d
                mask &= (mat[:, col] >= rng.get("min",0)) & (mat[:, col] <= rng.get("max",6))

    band_keys = ["1_10","11_20","21_30","31_40","41_45"]
    if req.band_enabled and req.band_filters:
        for bi, bkey in enumerate(band_keys):
            if bkey in req.band_filters:
                rng = req.band_filters[bkey]
                mask &= (mat[:, C_BAND+bi] >= rng.get("min",0)) & (mat[:, C_BAND+bi] <= rng.get("max",6))

    if req.palace_enabled and req.palace_filters:
        for pi in range(9):
            if str(pi) in req.palace_filters:
                rng = req.palace_filters[str(pi)]
                mask &= (mat[:, C_PAL+pi] >= rng.get("min",0)) & (mat[:, C_PAL+pi] <= rng.get("max",5))

    if req.paper_enabled and req.paper_filters:
        for ri in range(7):
            if str(ri) in req.paper_filters:
                rng = req.paper_filters[str(ri)]
                mask &= (mat[:, C_ROW+ri] >= rng.get("min",0)) & (mat[:, C_ROW+ri] <= rng.get("max",6))

    if req.regression_filters:
        for reg in req.regression_filters:
            reg_nums = np.array(reg.get("numbers",[]), dtype=np.int8)
            rmin = reg.get("min", 0)
            rmax = reg.get("max", 6)
            if len(reg_nums) > 0:
                cnt = np.isin(nums, reg_nums).sum(axis=1)
                mask &= (cnt >= rmin) & (cnt <= rmax)

    return {"count": int(np.sum(mask))}


@app.get("/api/health")
def health():
    ready = combos_mat is not None
    return {
        "status": "ready" if ready else "initializing",
        "total_combos": int(len(combos_mat)) if ready else 0
    }
