"""
lotto-filter-api / main.py

로또 1~45에서 6개를 뽑는 8,145,060가지 조합을 서버 시작 시
numpy 배열로 메모리에 올려두고, 필터 조건에 맞는 정확한 조합 수를 반환합니다.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from itertools import combinations
import time

app = FastAPI(title="Lotto Filter Count API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── 전역 배열 ──────────────────────────────────────────────────────
combos_matrix = None   # shape: (8145060, N_FEATURES) int8
combos_numbers = None  # shape: (8145060, 6) int8 — 번호 포함 여부용

PRIME_SET      = set([2,3,5,7,11,13,17,19,23,29,31,37,41,43])
COMPOSITE_SET  = set([4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,
                      27,28,30,32,33,34,35,36,38,39,40,42,44,45])
SQUARE_SET     = set([1,4,9,16,25,36])
TRIANGULAR_SET = set([1,3,6,10,15,21,28,36,45])
TWIN_SET       = set([11,22,33,44])

# 컬럼 인덱스
COL_SUM        = 0
COL_TAIL_SUM   = 1
COL_AC         = 2
COL_ODD        = 3
COL_HIGH       = 4
COL_PRIME      = 5
COL_COMPOSITE  = 6
COL_SQUARE     = 7
COL_TRIANGULAR = 8
COL_TWIN       = 9
COL_CONSEC     = 10
COL_TAIL_BASE  = 11   # +0~+9: 끝수별 개수
COL_BAND_BASE  = 21   # +0~+4: 번호대별
COL_PAL_BASE   = 26   # +0~+8: 9궁
COL_ROW_BASE   = 35   # +0~+6: 로또용지 행
N_FEATURES     = 42


def calc_ac(s):
    diffs = set()
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            diffs.add(s[j] - s[i])
    return len(diffs) - 5


def calc_consec(s):
    return sum(1 for i in range(len(s)-1) if s[i+1] - s[i] == 1)


def build_matrix():
    global combos_matrix, combos_numbers
    print("🔨 매트릭스 생성 시작 (int8, 메모리 최적화)...")
    t0 = time.time()

    all_combos = list(combinations(range(1, 46), 6))
    n = len(all_combos)

    # int8: -128~127 범위 → 로또 속성값 전부 커버 가능
    mat = np.zeros((n, N_FEATURES), dtype=np.int8)
    # 번호 6개를 직접 저장 (고정수/제외수 필터용)
    nums_mat = np.zeros((n, 6), dtype=np.int8)

    bands   = [(1,10),(11,20),(21,30),(31,40),(41,45)]
    palaces = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35),(36,40),(41,45)]

    for idx, nums in enumerate(all_combos):
        s = sorted(nums)
        nums_mat[idx] = s

        mat[idx, COL_SUM]        = sum(s)        # max 255 → int8 범위 초과!
        mat[idx, COL_TAIL_SUM]   = sum(x % 10 for x in s)
        mat[idx, COL_AC]         = calc_ac(s)
        mat[idx, COL_ODD]        = sum(1 for x in s if x % 2 == 1)
        mat[idx, COL_HIGH]       = sum(1 for x in s if x >= 23)
        mat[idx, COL_PRIME]      = sum(1 for x in s if x in PRIME_SET)
        mat[idx, COL_COMPOSITE]  = sum(1 for x in s if x in COMPOSITE_SET)
        mat[idx, COL_SQUARE]     = sum(1 for x in s if x in SQUARE_SET)
        mat[idx, COL_TRIANGULAR] = sum(1 for x in s if x in TRIANGULAR_SET)
        mat[idx, COL_TWIN]       = sum(1 for x in s if x in TWIN_SET)
        mat[idx, COL_CONSEC]     = calc_consec(s)
        for d in range(10):
            mat[idx, COL_TAIL_BASE + d] = sum(1 for x in s if x % 10 == d)
        for bi, (lo, hi) in enumerate(bands):
            mat[idx, COL_BAND_BASE + bi] = sum(1 for x in s if lo <= x <= hi)
        for pi, (lo, hi) in enumerate(palaces):
            mat[idx, COL_PAL_BASE + pi] = sum(1 for x in s if lo <= x <= hi)
        for ri in range(7):
            lo2 = ri * 7 + 1
            hi2 = lo2 + 6
            mat[idx, COL_ROW_BASE + ri] = sum(1 for x in s if lo2 <= x <= hi2)

    # 총합은 최대 255라 int8(-128~127) 초과 → int16 컬럼으로 별도 저장
    # 해결: 총합을 -128 오프셋으로 저장 (실제값 = 저장값 + 128)
    # 21~255 범위 → -107~127 (int8 OK)
    mat[:, COL_SUM] = (np.array([sum(c) for c in all_combos], dtype=np.int16) - 128).astype(np.int8)

    combos_matrix  = mat
    combos_numbers = nums_mat

    elapsed = time.time() - t0
    mem_mb = (mat.nbytes + nums_mat.nbytes) / 1024 / 1024
    print(f"✅ 완료: {n:,}개 조합, {elapsed:.1f}초, 메모리 {mem_mb:.1f}MB")


@app.on_event("startup")
def startup_event():
    build_matrix()


# ── Request 모델 ──────────────────────────────────────────────────
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


# ── 카운팅 API ────────────────────────────────────────────────────
@app.post("/api/count")
def count_combos(req: FilterRequest):
    if combos_matrix is None:
        return {"count": 0, "error": "초기화 중입니다. 잠시 후 다시 시도해주세요."}

    mat  = combos_matrix
    nums = combos_numbers
    mask = np.ones(len(mat), dtype=bool)

    # 1. 고정수 / 제외수
    if req.fixed or req.excluded:
        for n in req.fixed:
            if 1 <= n <= 45:
                mask &= np.any(nums == n, axis=1)
        for n in req.excluded:
            if 1 <= n <= 45:
                mask &= ~np.any(nums == n, axis=1)

    # 2. 총합 (오프셋 -128 적용)
    if req.total_sum_enabled:
        smin = np.int8(req.total_sum_min - 128)
        smax = np.int8(req.total_sum_max - 128)
        m = (mat[:, COL_SUM] >= smin) & (mat[:, COL_SUM] <= smax)
        if req.total_sum_excluded:
            excl = np.array([v - 128 for v in req.total_sum_excluded], dtype=np.int8)
            m &= ~np.isin(mat[:, COL_SUM], excl)
        mask &= m

    # 3. 끝수합
    if req.last_digit_sum_enabled:
        m = (mat[:, COL_TAIL_SUM] >= req.last_digit_sum_min) & \
            (mat[:, COL_TAIL_SUM] <= req.last_digit_sum_max)
        if req.last_digit_sum_excluded:
            m &= ~np.isin(mat[:, COL_TAIL_SUM],
                          np.array(req.last_digit_sum_excluded, dtype=np.int8))
        mask &= m

    # 4. AC값
    if req.ac_value_enabled:
        m = (mat[:, COL_AC] >= req.ac_value_min) & \
            (mat[:, COL_AC] <= req.ac_value_max)
        if req.ac_value_excluded:
            m &= ~np.isin(mat[:, COL_AC],
                          np.array(req.ac_value_excluded, dtype=np.int8))
        mask &= m

    # 5. 홀짝
    if req.odd_even_enabled and req.odd_even_counts:
        mask &= np.isin(mat[:, COL_ODD], np.array(req.odd_even_counts, dtype=np.int8))

    # 6. 저고
    if req.high_low_enabled and req.high_low_counts:
        mask &= np.isin(mat[:, COL_HIGH], np.array(req.high_low_counts, dtype=np.int8))

    # 7. 소수
    if req.prime_enabled and req.prime_counts:
        mask &= np.isin(mat[:, COL_PRIME], np.array(req.prime_counts, dtype=np.int8))

    # 8. 합성수
    if req.composite_enabled and req.composite_counts:
        mask &= np.isin(mat[:, COL_COMPOSITE], np.array(req.composite_counts, dtype=np.int8))

    # 9. 제곱수
    if req.square_enabled and req.square_counts:
        mask &= np.isin(mat[:, COL_SQUARE], np.array(req.square_counts, dtype=np.int8))

    # 10. 삼각수
    if req.triangular_enabled and req.triangular_counts:
        mask &= np.isin(mat[:, COL_TRIANGULAR], np.array(req.triangular_counts, dtype=np.int8))

    # 11. 쌍둥이수
    if req.twin_enabled and req.twin_counts:
        mask &= np.isin(mat[:, COL_TWIN], np.array(req.twin_counts, dtype=np.int8))

    # 12. 연번
    if req.consecutive_enabled and req.consecutive_counts:
        mask &= np.isin(mat[:, COL_CONSEC], np.array(req.consecutive_counts, dtype=np.int8))

    # 13. 끝수별
    if req.tail_digit_enabled and req.tail_digit_filters:
        for d_str, rng in req.tail_digit_filters.items():
            d = int(d_str)
            if 0 <= d <= 9:
                col = COL_TAIL_BASE + d
                mask &= (mat[:, col] >= rng.get("min", 0)) & \
                        (mat[:, col] <= rng.get("max", 6))

    # 14. 번호대별
    band_keys = ["1_10","11_20","21_30","31_40","41_45"]
    if req.band_enabled and req.band_filters:
        for bi, bkey in enumerate(band_keys):
            if bkey in req.band_filters:
                rng = req.band_filters[bkey]
                col = COL_BAND_BASE + bi
                mask &= (mat[:, col] >= rng.get("min", 0)) & \
                        (mat[:, col] <= rng.get("max", 6))

    # 15. 9궁
    if req.palace_enabled and req.palace_filters:
        for pi in range(9):
            if str(pi) in req.palace_filters:
                rng = req.palace_filters[str(pi)]
                col = COL_PAL_BASE + pi
                mask &= (mat[:, col] >= rng.get("min", 0)) & \
                        (mat[:, col] <= rng.get("max", 5))

    # 16. 로또용지 행
    if req.paper_enabled and req.paper_filters:
        for ri in range(7):
            if str(ri) in req.paper_filters:
                rng = req.paper_filters[str(ri)]
                col = COL_ROW_BASE + ri
                mask &= (mat[:, col] >= rng.get("min", 0)) & \
                        (mat[:, col] <= rng.get("max", 6))

    # 17. 회귀 분석
    if req.regression_filters:
        for reg in req.regression_filters:
            reg_nums = np.array(reg.get("numbers", []), dtype=np.int8)
            rmin = reg.get("min", 0)
            rmax = reg.get("max", 6)
            if len(reg_nums) > 0:
                cnt = np.sum(np.isin(nums, reg_nums), axis=1)
                mask &= (cnt >= rmin) & (cnt <= rmax)

    return {"count": int(np.sum(mask))}


@app.get("/api/health")
def health():
    ready = combos_matrix is not None
    return {
        "status": "ready" if ready else "initializing",
        "total_combos": int(len(combos_matrix)) if ready else 0
    }
