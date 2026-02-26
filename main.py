"""
lotto-filter-api / main.py (v2 - 완전 numpy 벡터 연산, 메모리 최적화)
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

# 전역 변수
combos_mat = None   # (8145060, N_FEATURES) int8
combos_nums = None  # (8145060, 6) int8

PRIME_SET      = frozenset([2,3,5,7,11,13,17,19,23,29,31,37,41,43])
COMPOSITE_SET  = frozenset([4,6,8,9,10,12,14,15,16,18,20,21,22,24,25,
                             26,27,28,30,32,33,34,35,36,38,39,40,42,44,45])
SQUARE_SET     = frozenset([1,4,9,16,25,36])
TRIANGULAR_SET = frozenset([1,3,6,10,15,21,28,36,45])
TWIN_SET       = frozenset([11,22,33,44])

# 컬럼 인덱스
C_SUM=0; C_TAIL=1; C_AC=2; C_ODD=3; C_HIGH=4
C_PRIME=5; C_COMP=6; C_SQ=7; C_TRI=8; C_TWIN=9; C_CONS=10
C_TDIG=11  # +0~9: 끝수별
C_BAND=21  # +0~4: 번호대
C_PAL=26   # +0~8: 9궁
C_ROW=35   # +0~6: 로또용지행
N_FEAT=42


def build_matrix():
    global combos_mat, combos_nums
    print("🔨 매트릭스 생성 시작 (numpy 벡터 연산)...")
    t0 = time.time()

    # 1. 전체 조합을 numpy 배열로 생성
    all_c = np.array(list(combinations(range(1, 46), 6)), dtype=np.int8)  # (8145060, 6)
    n = len(all_c)
    print(f"   조합 배열 생성: {all_c.nbytes/1024/1024:.1f}MB")

    mat = np.zeros((n, N_FEAT), dtype=np.int8)

    # 2. 벡터 연산으로 한번에 계산
    # 총합 (최대255 → int16로 계산 후 -128 오프셋 저장)
    total_sum = all_c.astype(np.int16).sum(axis=1)
    mat[:, C_SUM] = (total_sum - 128).astype(np.int8)

    # 끝수합
    mat[:, C_TAIL] = (all_c % 10).sum(axis=1).astype(np.int8)

    # 홀수 개수
    mat[:, C_ODD] = (all_c % 2).sum(axis=1).astype(np.int8)

    # 고번호(23~45) 개수
    mat[:, C_HIGH] = (all_c >= 23).sum(axis=1).astype(np.int8)

    # 소수 개수
    prime_arr = np.array(sorted(PRIME_SET), dtype=np.int8)
    mat[:, C_PRIME] = np.isin(all_c, prime_arr).sum(axis=1).astype(np.int8)

    # 합성수 개수
    comp_arr = np.array(sorted(COMPOSITE_SET), dtype=np.int8)
    mat[:, C_COMP] = np.isin(all_c, comp_arr).sum(axis=1).astype(np.int8)

    # 제곱수 개수
    sq_arr = np.array(sorted(SQUARE_SET), dtype=np.int8)
    mat[:, C_SQ] = np.isin(all_c, sq_arr).sum(axis=1).astype(np.int8)

    # 삼각수 개수
    tri_arr = np.array(sorted(TRIANGULAR_SET), dtype=np.int8)
    mat[:, C_TRI] = np.isin(all_c, tri_arr).sum(axis=1).astype(np.int8)

    # 쌍둥이수 개수
    twin_arr = np.array(sorted(TWIN_SET), dtype=np.int8)
    mat[:, C_TWIN] = np.isin(all_c, twin_arr).sum(axis=1).astype(np.int8)

    # 연번 쌍 개수 (인접 열 차이가 1인 쌍)
    diffs = np.diff(all_c.astype(np.int16), axis=1)  # (n, 5)
    mat[:, C_CONS] = (diffs == 1).sum(axis=1).astype(np.int8)

    # AC값 (조합 내 모든 차이값의 고유 개수 - 5)
    # 벡터화가 복잡하므로 청크 단위 처리
    print("   AC값 계산 중...")
    chunk = 100000
    for i in range(0, n, chunk):
        batch = all_c[i:i+chunk].astype(np.int16)
        ac_vals = []
        for row in batch:
            s = sorted(row)
            dset = set()
            for a in range(6):
                for b in range(a+1, 6):
                    dset.add(int(s[b]-s[a]))
            ac_vals.append(len(dset) - 5)
        mat[i:i+chunk, C_AC] = np.array(ac_vals, dtype=np.int8)

    # 끝수별 개수
    for d in range(10):
        mat[:, C_TDIG+d] = (all_c % 10 == d).sum(axis=1).astype(np.int8)

    # 번호대별 개수
    bands = [(1,10),(11,20),(21,30),(31,40),(41,45)]
    for bi, (lo, hi) in enumerate(bands):
        mat[:, C_BAND+bi] = ((all_c >= lo) & (all_c <= hi)).sum(axis=1).astype(np.int8)

    # 9궁 구간별 개수
    palaces = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35),(36,40),(41,45)]
    for pi, (lo, hi) in enumerate(palaces):
        mat[:, C_PAL+pi] = ((all_c >= lo) & (all_c <= hi)).sum(axis=1).astype(np.int8)

    # 로또용지 행별 개수
    for ri in range(7):
        lo2 = ri*7+1; hi2 = lo2+6
        mat[:, C_ROW+ri] = ((all_c >= lo2) & (all_c <= hi2)).sum(axis=1).astype(np.int8)

    combos_mat = mat
    combos_nums = all_c

    total_mb = (mat.nbytes + all_c.nbytes) / 1024 / 1024
    print(f"✅ 완료: {n:,}개, {time.time()-t0:.1f}초, 총 {total_mb:.1f}MB")


@app.on_event("startup")
def startup_event():
    build_matrix()


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

    mat = combos_mat
    nums = combos_nums
    mask = np.ones(len(mat), dtype=bool)

    # 고정수/제외수
    if req.fixed or req.excluded:
        for n in req.fixed:
            if 1 <= n <= 45:
                mask &= np.any(nums == np.int8(n), axis=1)
        for n in req.excluded:
            if 1 <= n <= 45:
                mask &= ~np.any(nums == np.int8(n), axis=1)

    # 총합
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
        m = (mat[:, C_AC] >= req.ac_value_min) & (mat[:, C_AC] <= req.ac_value_max)
        if req.ac_value_excluded:
            m &= ~np.isin(mat[:, C_AC], np.array(req.ac_value_excluded, dtype=np.int8))
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
                col = C_BAND + bi
                mask &= (mat[:, col] >= rng.get("min",0)) & (mat[:, col] <= rng.get("max",6))

    if req.palace_enabled and req.palace_filters:
        for pi in range(9):
            if str(pi) in req.palace_filters:
                rng = req.palace_filters[str(pi)]
                col = C_PAL + pi
                mask &= (mat[:, col] >= rng.get("min",0)) & (mat[:, col] <= rng.get("max",5))

    if req.paper_enabled and req.paper_filters:
        for ri in range(7):
            if str(ri) in req.paper_filters:
                rng = req.paper_filters[str(ri)]
                col = C_ROW + ri
                mask &= (mat[:, col] >= rng.get("min",0)) & (mat[:, col] <= rng.get("max",6))

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
