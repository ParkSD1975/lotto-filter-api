"""
lotto-filter-api / main.py

로또 1~45에서 6개를 뽑는 8,145,060가지 조합을 서버 시작 시
numpy 배열로 메모리에 올려두고, 필터 조건에 맞는 정확한 조합 수를 반환합니다.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
from itertools import combinations
import time

app = FastAPI(title="Lotto Filter Count API")

# CORS 설정 (프론트엔드 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 후 실제 도메인으로 교체 권장
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── 전역 numpy 배열 (서버 시작 시 1회 계산) ──────────────────────
# 각 조합의 속성을 int16으로 저장 (메모리 최소화)
combos_matrix = None   # shape: (8145060, N_FEATURES)

PRIME_SET  = set([2,3,5,7,11,13,17,19,23,29,31,37,41,43])
COMPOSITE_SET = set([4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,26,
                     27,28,30,32,33,34,35,36,38,39,40,42,44,45])
SQUARE_SET    = set([1,4,9,16,25,36])
TRIANGULAR_SET= set([1,3,6,10,15,21,28,36,45])
TWIN_SET      = set([11,22,33,44])

# 컬럼 인덱스 상수
COL_SUM        = 0
COL_TAIL_SUM   = 1
COL_AC         = 2
COL_ODD        = 3   # 홀수 개수
COL_HIGH       = 4   # 고번호(23~45) 개수
COL_PRIME      = 5
COL_COMPOSITE  = 6
COL_SQUARE     = 7
COL_TRIANGULAR = 8
COL_TWIN       = 9
COL_CONSEC     = 10  # 연번 쌍 개수
# COL 11~20: 끝수(0~9)별 개수
COL_TAIL_DIGIT_BASE = 11   # COL_TAIL_DIGIT_BASE + d = 끝수 d의 개수
# COL 21~29: 번호대별 개수 (1~10, 11~20, 21~30, 31~40, 41~45)
COL_BAND_BASE  = 21  # +0~+4
# COL 26~34: 9궁 구간별 개수 (1~5, 6~10, 11~15, 16~20, 21~25, 26~30, 31~35, 36~40, 41~45)
COL_PALACE_BASE = 26  # +0~+8
# COL 35~41: 로또용지 행별 개수 (행1~행7, 각 행 7개)
COL_PAPER_ROW_BASE = 35  # +0~+6

N_FEATURES = 42


def calc_ac(nums):
    """AC값 계산"""
    nums_sorted = sorted(nums)
    diffs = set()
    for i in range(len(nums_sorted)):
        for j in range(i+1, len(nums_sorted)):
            diffs.add(nums_sorted[j] - nums_sorted[i])
    return len(diffs) - 5


def calc_consec(nums):
    """연번 쌍 개수"""
    s = sorted(nums)
    return sum(1 for i in range(len(s)-1) if s[i+1] - s[i] == 1)


def build_matrix():
    """서버 시작 시 8,145,060개 조합 전체 속성 계산"""
    global combos_matrix
    print("🔨 조합 매트릭스 생성 시작...")
    t0 = time.time()

    all_combos = list(combinations(range(1, 46), 6))
    n = len(all_combos)  # 8,145,060
    mat = np.zeros((n, N_FEATURES), dtype=np.int16)

    for idx, nums in enumerate(all_combos):
        s = sorted(nums)
        mat[idx, COL_SUM]       = sum(s)
        mat[idx, COL_TAIL_SUM]  = sum(x % 10 for x in s)
        mat[idx, COL_AC]        = calc_ac(s)
        mat[idx, COL_ODD]       = sum(1 for x in s if x % 2 == 1)
        mat[idx, COL_HIGH]      = sum(1 for x in s if x >= 23)
        mat[idx, COL_PRIME]     = sum(1 for x in s if x in PRIME_SET)
        mat[idx, COL_COMPOSITE] = sum(1 for x in s if x in COMPOSITE_SET)
        mat[idx, COL_SQUARE]    = sum(1 for x in s if x in SQUARE_SET)
        mat[idx, COL_TRIANGULAR]= sum(1 for x in s if x in TRIANGULAR_SET)
        mat[idx, COL_TWIN]      = sum(1 for x in s if x in TWIN_SET)
        mat[idx, COL_CONSEC]    = calc_consec(s)
        # 끝수별 개수
        for d in range(10):
            mat[idx, COL_TAIL_DIGIT_BASE + d] = sum(1 for x in s if x % 10 == d)
        # 번호대별 개수
        bands = [(1,10),(11,20),(21,30),(31,40),(41,45)]
        for bi, (lo, hi) in enumerate(bands):
            mat[idx, COL_BAND_BASE + bi] = sum(1 for x in s if lo <= x <= hi)
        # 9궁 구간별 개수
        palaces = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35),(36,40),(41,45)]
        for pi, (lo, hi) in enumerate(palaces):
            mat[idx, COL_PALACE_BASE + pi] = sum(1 for x in s if lo <= x <= hi)
        # 로또용지 행별 개수 (1행: 1~7, 2행: 8~14, ...)
        for ri in range(7):
            lo2 = ri * 7 + 1
            hi2 = lo2 + 6
            mat[idx, COL_PAPER_ROW_BASE + ri] = sum(1 for x in s if lo2 <= x <= hi2)

    combos_matrix = mat
    elapsed = time.time() - t0
    print(f"✅ 매트릭스 생성 완료: {n:,}개 조합, {elapsed:.1f}초 소요")
    print(f"   메모리 사용: {mat.nbytes / 1024 / 1024:.1f} MB")


# 앱 시작 시 매트릭스 빌드
@app.on_event("startup")
def startup_event():
    build_matrix()


# ── Request 모델 ──────────────────────────────────────────────────
class FilterRequest(BaseModel):
    # 고정수/제외수
    fixed: list[int] = []
    excluded: list[int] = []

    # 총합
    total_sum_enabled: bool = False
    total_sum_min: int = 21
    total_sum_max: int = 255
    total_sum_excluded: list[int] = []

    # 끝수합
    last_digit_sum_enabled: bool = False
    last_digit_sum_min: int = 2
    last_digit_sum_max: int = 52
    last_digit_sum_excluded: list[int] = []

    # AC값
    ac_value_enabled: bool = False
    ac_value_min: int = 0
    ac_value_max: int = 10
    ac_value_excluded: list[int] = []

    # 홀짝 패턴 (허용할 홀수 개수 배열, 빈 배열=전체허용)
    odd_even_enabled: bool = False
    odd_even_counts: list[int] = []

    # 저고 패턴 (허용할 고번호 개수 배열)
    high_low_enabled: bool = False
    high_low_counts: list[int] = []

    # 소수 개수
    prime_enabled: bool = False
    prime_counts: list[int] = []

    # 합성수 개수
    composite_enabled: bool = False
    composite_counts: list[int] = []

    # 제곱수 개수
    square_enabled: bool = False
    square_counts: list[int] = []

    # 삼각수 개수
    triangular_enabled: bool = False
    triangular_counts: list[int] = []

    # 쌍둥이수 개수
    twin_enabled: bool = False
    twin_counts: list[int] = []

    # 연번 개수
    consecutive_enabled: bool = False
    consecutive_counts: list[int] = []

    # 끝수별 개수 (digits: {"0": {"min":0,"max":6}, ...})
    tail_digit_enabled: bool = False
    tail_digit_filters: dict = {}

    # 번호대별 개수 (bands: {"1_10": {"min":0,"max":6}, ...})
    band_enabled: bool = False
    band_filters: dict = {}

    # 9궁별 개수
    palace_enabled: bool = False
    palace_filters: dict = {}

    # 로또용지 행별 개수
    paper_enabled: bool = False
    paper_filters: dict = {}

    # 회귀 분석 (step: 2~200, 각 step의 번호 목록과 min/max 포함 개수)
    regression_filters: list[dict] = []
    # 예: [{"numbers": [3,7,12,25,38,41], "min": 0, "max": 2}, ...]


# ── 카운팅 API ────────────────────────────────────────────────────
@app.post("/api/count")
def count_combos(req: FilterRequest):
    if combos_matrix is None:
        return {"count": 0, "error": "매트릭스 초기화 중입니다. 잠시 후 다시 시도해주세요."}

    mat = combos_matrix
    mask = np.ones(len(mat), dtype=bool)

    # 1. 고정수 / 제외수 처리
    # 고정수 f개가 포함된 조합만 허용
    # → 전체 번호 풀에서 계산하므로, 고정수/제외수는 별도 nCr로만 처리
    # (매트릭스는 전체 45개 기준이므로, 고정수/제외수 번호를 마스킹)
    fixed_set = set(req.fixed)
    excluded_set = set(req.excluded)

    if fixed_set or excluded_set:
        # 고정수가 모두 포함되고, 제외수가 하나도 없는 조합만
        # 전체 조합을 순회하지 않고 numpy로 처리
        # 각 번호의 포함 여부를 원-핫으로 저장하지 않았으므로
        # 별도 번호 매트릭스 필요 → 간단히 재구성
        # (성능상 번호 매트릭스를 따로 만드는 게 좋지만, 여기서는 Python loop 최소화)
        # 고정수/제외수가 있으면 별도 필터링 함수 사용
        mask = _apply_basket_mask(mask, req.fixed, req.excluded)

    # 2. 총합
    if req.total_sum_enabled:
        m = (mat[:, COL_SUM] >= req.total_sum_min) & (mat[:, COL_SUM] <= req.total_sum_max)
        if req.total_sum_excluded:
            excl = np.array(req.total_sum_excluded, dtype=np.int16)
            m &= ~np.isin(mat[:, COL_SUM], excl)
        mask &= m

    # 3. 끝수합
    if req.last_digit_sum_enabled:
        m = (mat[:, COL_TAIL_SUM] >= req.last_digit_sum_min) & (mat[:, COL_TAIL_SUM] <= req.last_digit_sum_max)
        if req.last_digit_sum_excluded:
            excl = np.array(req.last_digit_sum_excluded, dtype=np.int16)
            m &= ~np.isin(mat[:, COL_TAIL_SUM], excl)
        mask &= m

    # 4. AC값
    if req.ac_value_enabled:
        m = (mat[:, COL_AC] >= req.ac_value_min) & (mat[:, COL_AC] <= req.ac_value_max)
        if req.ac_value_excluded:
            excl = np.array(req.ac_value_excluded, dtype=np.int16)
            m &= ~np.isin(mat[:, COL_AC], excl)
        mask &= m

    # 5. 홀짝
    if req.odd_even_enabled and req.odd_even_counts:
        mask &= np.isin(mat[:, COL_ODD], np.array(req.odd_even_counts, dtype=np.int16))

    # 6. 저고
    if req.high_low_enabled and req.high_low_counts:
        mask &= np.isin(mat[:, COL_HIGH], np.array(req.high_low_counts, dtype=np.int16))

    # 7. 소수
    if req.prime_enabled and req.prime_counts:
        mask &= np.isin(mat[:, COL_PRIME], np.array(req.prime_counts, dtype=np.int16))

    # 8. 합성수
    if req.composite_enabled and req.composite_counts:
        mask &= np.isin(mat[:, COL_COMPOSITE], np.array(req.composite_counts, dtype=np.int16))

    # 9. 제곱수
    if req.square_enabled and req.square_counts:
        mask &= np.isin(mat[:, COL_SQUARE], np.array(req.square_counts, dtype=np.int16))

    # 10. 삼각수
    if req.triangular_enabled and req.triangular_counts:
        mask &= np.isin(mat[:, COL_TRIANGULAR], np.array(req.triangular_counts, dtype=np.int16))

    # 11. 쌍둥이수
    if req.twin_enabled and req.twin_counts:
        mask &= np.isin(mat[:, COL_TWIN], np.array(req.twin_counts, dtype=np.int16))

    # 12. 연번
    if req.consecutive_enabled and req.consecutive_counts:
        mask &= np.isin(mat[:, COL_CONSEC], np.array(req.consecutive_counts, dtype=np.int16))

    # 13. 끝수별 개수
    if req.tail_digit_enabled and req.tail_digit_filters:
        for d_str, rng in req.tail_digit_filters.items():
            d = int(d_str)
            if 0 <= d <= 9:
                col = COL_TAIL_DIGIT_BASE + d
                mask &= (mat[:, col] >= rng.get("min", 0)) & (mat[:, col] <= rng.get("max", 6))

    # 14. 번호대별
    band_keys = ["1_10", "11_20", "21_30", "31_40", "41_45"]
    if req.band_enabled and req.band_filters:
        for bi, bkey in enumerate(band_keys):
            if bkey in req.band_filters:
                rng = req.band_filters[bkey]
                col = COL_BAND_BASE + bi
                mask &= (mat[:, col] >= rng.get("min", 0)) & (mat[:, col] <= rng.get("max", 6))

    # 15. 9궁
    if req.palace_enabled and req.palace_filters:
        for pi in range(9):
            pkey = str(pi)
            if pkey in req.palace_filters:
                rng = req.palace_filters[pkey]
                col = COL_PALACE_BASE + pi
                mask &= (mat[:, col] >= rng.get("min", 0)) & (mat[:, col] <= rng.get("max", 5))

    # 16. 로또용지 행
    if req.paper_enabled and req.paper_filters:
        for ri in range(7):
            rkey = str(ri)
            if rkey in req.paper_filters:
                rng = req.paper_filters[rkey]
                col = COL_PAPER_ROW_BASE + ri
                mask &= (mat[:, col] >= rng.get("min", 0)) & (mat[:, col] <= rng.get("max", 6))

    # 17. 회귀 분석
    if req.regression_filters:
        for reg in req.regression_filters:
            nums = reg.get("numbers", [])
            rmin = reg.get("min", 0)
            rmax = reg.get("max", 6)
            if nums:
                # 해당 조합에서 reg 번호들이 몇 개 포함되는지
                # numpy isin으로 처리
                reg_arr = np.array(nums, dtype=np.int16)
                # 이 부분은 번호 매트릭스가 필요 → 별도 함수
                mask &= _calc_regression_mask(mask, reg_arr, rmin, rmax)

    count = int(np.sum(mask))
    return {"count": count}


@app.get("/api/health")
def health():
    ready = combos_matrix is not None
    return {
        "status": "ready" if ready else "initializing",
        "total_combos": int(len(combos_matrix)) if ready else 0
    }


# ── 헬퍼: 고정수/제외수 마스크 ────────────────────────────────────
_basket_matrix = None  # shape: (8145060, 45) bool — 번호 포함 여부

def _get_basket_matrix():
    """번호 포함 여부 bool 매트릭스 (lazy init)"""
    global _basket_matrix
    if _basket_matrix is not None:
        return _basket_matrix
    print("🔨 번호 매트릭스 생성 중...")
    all_combos = list(combinations(range(1, 46), 6))
    bmat = np.zeros((len(all_combos), 45), dtype=bool)
    for idx, nums in enumerate(all_combos):
        for n in nums:
            bmat[idx, n-1] = True
    _basket_matrix = bmat
    print("✅ 번호 매트릭스 완료")
    return bmat

def _apply_basket_mask(mask, fixed, excluded):
    bmat = _get_basket_matrix()
    new_mask = mask.copy()
    for n in fixed:
        if 1 <= n <= 45:
            new_mask &= bmat[:, n-1]
    for n in excluded:
        if 1 <= n <= 45:
            new_mask &= ~bmat[:, n-1]
    return new_mask

def _calc_regression_mask(mask, reg_arr, rmin, rmax):
    """회귀 번호들이 rmin~rmax개 포함된 조합 마스크"""
    bmat = _get_basket_matrix()
    # reg_arr에 있는 번호들의 포함 개수 합산
    reg_cols = reg_arr - 1  # 0-indexed
    reg_cols = reg_cols[(reg_cols >= 0) & (reg_cols < 45)]
    count_per_combo = bmat[:, reg_cols].sum(axis=1)
    return (count_per_combo >= rmin) & (count_per_combo <= rmax)
