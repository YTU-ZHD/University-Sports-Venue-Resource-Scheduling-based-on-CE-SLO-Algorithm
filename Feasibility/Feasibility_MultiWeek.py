from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
from concurrent.futures import ProcessPoolExecutor
import argparse
import time

"""
命令：
python Feasibility_MultiWeek_version_1.3_Gemini.py --csv ../Init_Data/DataSet_123.csv --seed 2007
python Feasibility_MultiWeek_version_1.3_Gemini.py --csv ../Init_Data/DataSet_weeks2_123.csv --seed 2007
python Feasibility_MultiWeek_version_1.3_Gemini.py --csv ../Init_Data/DataSet_weeks3_123.csv --seed 2007
python Feasibility_MultiWeek_version_1.3_Gemini.py --csv ../Init_Data/DataSet_weeks4_123.csv --seed 2007
python Feasibility_MultiWeek_version_1.3_Gemini.py --csv ../Init_Data/DataSet_weeks5_123.csv --seed 2007

"""

# 轻量进度条
def _print_progress(prefix: str, i: int, total: int, width: int = 30):
    total = max(1, int(total));
    i = min(max(0, int(i)), total)
    filled = int(width * i / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix} [{bar}] {i}/{total}", end="", flush=True)
    if i >= total:
        print()


# ===================== 常量与基础工具 =====================
DAYS = 7
COURTS = 3
SLOTS_PER_DAY = 12
TOTAL_SLOTS = DAYS * COURTS * SLOTS_PER_DAY  # 252

# 槽位容量（除 FREE 外均为 1）
CAPACITY_NONFREE = 1
CAPACITY_FREE = 10

# 请求类型枚举
HARD, TEACHING, SOFT, CLUB, FREE = "HARD", "TEACHING", "SOFT", "CLUB", "FREE"
TYPE_ORDER = [HARD, TEACHING, SOFT, CLUB, FREE]  # 优先级从高到低

# 必成类（强制排上）
MUST_HAVE = {HARD, TEACHING, SOFT}

# 允许的时间窗口（按文档说明）
ALLOWED_DAYS = {
    HARD: list(range(0, 7)),  # 周一-周日（必须按期望时间）
    SOFT: list(range(0, 7)),  # 周一-周日（可调整）
    TEACHING: list(range(0, 5)),  # 周一-周五
    CLUB: list(range(0, 7)),  # 周一-周日
    FREE: [5, 6],  # 周六-周日
}

DAY_NAME_TO_IDX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6
}


# ========== 槽位编码/解码 ==========

def enc(day: int, court: int, start: int) -> int:
    """ 将 (day, court, start_slot) 编码为单整数基因值（1..252 作为“已分配”），0 表示未分配。"""
    return day * (COURTS * SLOTS_PER_DAY) + court * SLOTS_PER_DAY + start + 1  # +1 规避 0


def dec(code: int) -> Tuple[int, int, int]:
    """ 解码为 (day, court, start_slot)；code=0 表示未分配（抛异常由调用方处理或特判）。"""
    idx = code - 1
    day = idx // (COURTS * SLOTS_PER_DAY)
    rem = idx % (COURTS * SLOTS_PER_DAY)
    court = rem // SLOTS_PER_DAY
    start = rem % SLOTS_PER_DAY
    return day, court, start


# ===================== 数据结构 =====================
@dataclass
class Request:
    rid: int
    rtype: str  # 五类之一
    duration: int  # 1-4
    pref_day: int  # 0..DAYS-1（跨周整数天索引）
    pref_start: int  # 0-11
    pref_court: Optional[int]  # 0-2 或 None（若 CSV 无场馆偏好）

    def __hash__(self):
        return hash(self.rid)


@dataclass
class Individual:
    genes: List[int]  # 长度 R，基因为编码后的起始槽位（0 表示未分配）
    fitness: float = 0.0
    comp: Dict[str, float] = field(default_factory=dict)  # 各分量（util/adj/fair/pen）

    def clone(self) -> 'Individual':
        return Individual(self.genes.copy(), self.fitness, self.comp.copy())


# ===================== 请求生成/加载 =====================

def make_demo_requests(seed: int = 42) -> List[Request]:
    """按文档“实验例子”生成一周的请求集（总约 170 条）。"""
    rng = random.Random(seed)
    reqs: List[Request] = []

    def add(n, rtype, dur_rng):
        nonlocal reqs
        for _ in range(n):
            d = rng.choice(ALLOWED_DAYS[rtype])
            dur = rng.choice(dur_rng)
            max_start = SLOTS_PER_DAY - dur
            start = rng.randint(0, max_start)
            court = rng.randint(0, COURTS - 1)
            # 演示集默认都给一个轻偏好场馆
            reqs.append(Request(len(reqs), rtype, dur, d, start, court))

    add(8, HARD, [3, 4])
    add(12, SOFT, [3, 4])
    add(60, TEACHING, [2])
    add(40, CLUB, [1, 2, 3])
    add(50, FREE, [1])
    return reqs


def _coerce_day(x) -> int:
    if isinstance(x, int):
        return x
    s = str(x).strip().lower()
    if s in DAY_NAME_TO_IDX:
        return DAY_NAME_TO_IDX[s]
    raise ValueError(f"Unrecognized day: {x}")


def load_requests_from_csv(path: str) -> List[Request]:
    """ 兼容两种 CSV：
    A) 你项目的 Init.py 产物：RequestID, Type, DesiredDay, DesiredSlot, Duration
       - DesiredDay: Monday..Sunday；DesiredSlot: 1..12（本函数转为 0..11）
       - 无 preferred_court → 设为 None；对 HARD 解释为“场馆不限”。
    B) 研究版自定义：id,type,preferred_day,preferred_start,preferred_court,duration
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        idx = {h.lower(): i for i, h in enumerate(header)}
        reqs: List[Request] = []

        # 方案 A（Init.py）
        if {"requestid", "type", "desiredday", "desiredslot", "duration"} <= set(idx.keys()):
            tmap = {
                "hard game": HARD,
                "soft game": SOFT,
                "teaching": TEACHING,
                "student club": CLUB,
                "free exercise": FREE,
            }
            for row in reader:
                rtype_name = row[idx["type"]].strip().lower()
                rtype = tmap[rtype_name]
                d = _coerce_day(row[idx["desiredday"]])
                s = int(row[idx["desiredslot"]]) - 1  # 1→0 起算
                dur = int(row[idx["duration"]])
                reqs.append(Request(len(reqs), rtype, dur, d, s, None))
            return reqs

        # 方案 B（研究版）
        needed_core = ["id", "type", "preferred_day", "preferred_start", "duration"]
        for key in needed_core:
            if key not in idx:
                raise ValueError(f"CSV missing column: {key}")
        has_pc = ("preferred_court" in idx)

        for row in reader:
            rtype = row[idx["type"]].strip().upper()
            pd = int(row[idx["preferred_day"]])
            ps = int(row[idx["preferred_start"]])
            dur = int(row[idx["duration"]])
            if has_pc:
                pc_raw = row[idx["preferred_court"]].strip()
                pc = int(pc_raw) if pc_raw != "" else None
            else:
                pc = None
            reqs.append(Request(len(reqs), rtype, dur, pd, ps, pc))
        return reqs


# ===================== 调度状态（占用与统计） =====================
class Occupancy:
    """ 记录槽位占用计数，并提供可行性检查/落位/迁移。
    修正：区分 FREE 与非 FREE 的占用，避免 FREE 与非 FREE 叠加；
    - 非 FREE：与任何占用（FREE 或非 FREE）互斥，容量上限 1；
    - FREE：不得与任何非 FREE 同槽位叠加，但可与 FREE 叠加至 CAPACITY_FREE。
    """

    def __init__(self):
        # day -> court -> slot -> count
        self.nonfree = [[[0 for _ in range(SLOTS_PER_DAY)] for _ in range(COURTS)] for _ in range(DAYS)]
        self.free = [[[0 for _ in range(SLOTS_PER_DAY)] for _ in range(COURTS)] for _ in range(DAYS)]
        # 学生社团/自由锻炼累计时段数（公平性度量）
        self.club_slots = 0
        self.free_slots = 0

    def capacity_of(self, rtype: str) -> int:
        return CAPACITY_FREE if rtype == FREE else CAPACITY_NONFREE

    def can_place(self, r: Request, day: int, court: int, start: int) -> bool:
        if day not in ALLOWED_DAYS[r.rtype]:
            return False
        if start + r.duration > SLOTS_PER_DAY:
            return False
        if r.rtype == FREE:
            # FREE 不得与非 FREE 叠加；FREE 自身可叠加到 CAPACITY_FREE
            for t in range(start, start + r.duration):
                if self.nonfree[day][court][t] > 0:
                    return False
                if self.free[day][court][t] >= CAPACITY_FREE:
                    return False
            return True
        else:
            # 非 FREE：与任何占用互斥（包括 FREE）
            for t in range(start, start + r.duration):
                if self.nonfree[day][court][t] > 0:
                    return False
                if self.free[day][court][t] > 0:
                    return False
            return True

    def place(self, r: Request, day: int, court: int, start: int):
        if r.rtype == FREE:
            for t in range(start, start + r.duration):
                self.free[day][court][t] += 1
            self.free_slots += r.duration
        else:
            for t in range(start, start + r.duration):
                self.nonfree[day][court][t] += 1
            if r.rtype == CLUB:
                self.club_slots += r.duration

    def remove(self, r: Request, day: int, court: int, start: int):
        if r.rtype == FREE:
            for t in range(start, start + r.duration):
                self.free[day][court][t] -= 1
            self.free_slots -= r.duration
        else:
            for t in range(start, start + r.duration):
                self.nonfree[day][court][t] -= 1
            if r.rtype == CLUB:
                self.club_slots -= r.duration


# ===================== 候选起点生成与排序 =====================

def candidate_starts(r: Request) -> List[Tuple[int, int, int]]:
    """枚举不跨日的可选起点（仅做静态时间窗筛选，不考虑占用）。"""
    cands = []
    days = ALLOWED_DAYS[r.rtype]
    for d in days:
        for court in range(COURTS):
            max_start = SLOTS_PER_DAY - r.duration
            for s in range(0, max_start + 1):
                cands.append((d, court, s))
    return cands


def sort_candidates(r: Request, cands: List[Tuple[int, int, int]], occ: Occupancy) -> List[Tuple[int, int, int]]:
    """按优先级：偏好同日/近时段、次偏好场馆（若存在），对 CLUB/FREE 同分时轻微公平性偏置。"""

    def key(c):
        d, court, s = c
        day_gap = abs(d - r.pref_day)
        slot_gap = abs(s - r.pref_start)
        court_bias = 0 if (r.pref_court is None or court == r.pref_court) else 1
        return (day_gap, slot_gap, court_bias)

    return sorted(cands, key=key)


# ===================== 适应度函数 =====================
@dataclass
class FitnessParams:
    w_util: float = 0.35
    w_adj: float = 0.35
    w_fair: float = 0.2
    w_pen: float = 0.1
    # 异日/同日打分参数
    adj_same_max: float = 1.0
    adj_cross_day_max: float = 0.9
    day_alpha: float = 0.25  # 天差惩罚系数
    slot_beta: float = 0.05  # 时段差惩罚系数


class Fitness:
    def __init__(self, reqs: List[Request], params: FitnessParams):
        self.reqs = reqs
        self.R = len(reqs)
        self.p = params

        # --- 新增：预计算 CLUB 和 FREE 的总需求时长，用于计算“满足率公平性”（对应新公式6） ---
        self.total_club_demand = sum(r.duration for r in reqs if r.rtype == CLUB)
        self.total_free_demand = sum(r.duration for r in reqs if r.rtype == FREE)

    def eval(self, indiv: Individual) -> Individual:
        # 统计：利用率、时间调整得分、公平性、惩罚
        adj_score_total = 0.0
        unallocated_rigid_slots = 0.0  # 记录未分配的刚性需求总时长（对应新公式7分子）
        club_slots = 0
        free_slots = 0

        # 为了得到真实“利用率≤1”，按槽位维度统计是否被占用（而不是“累计时段数”）
        used = [[[0 for _ in range(SLOTS_PER_DAY)] for _ in range(COURTS)] for _ in range(DAYS)]

        for i, gene in enumerate(indiv.genes):
            r = self.reqs[i]
            if gene == 0:
                if r.rtype in MUST_HAVE:
                    unallocated_rigid_slots += r.duration  # 累计未被分配的刚性时长
                continue
            d, c, s = dec(gene)
            # 标记占用：只要该槽位有任何活动（FREE 或非 FREE）就算 1
            for t in range(s, s + r.duration):
                used[d][c][t] = 1

            # 时间调整得分
            if d == r.pref_day:
                gap = abs(s - r.pref_start)
                same = max(self.p.adj_same_max - self.p.slot_beta * gap, 0.0)
                adj_score_total += same
            else:
                day_gap = abs(d - r.pref_day)
                slot_gap = abs(s - r.pref_start)
                cross = max(self.p.adj_cross_day_max - self.p.day_alpha * day_gap - self.p.slot_beta * slot_gap, 0.0)
                adj_score_total += cross

            # 统计CLUB和FREE已分配的实际时长
            if r.rtype == CLUB:
                club_slots += r.duration
            elif r.rtype == FREE:
                free_slots += r.duration

        # 1. 利用率 (Util)
        used_slots = sum(used[d][c][t] for d in range(DAYS) for c in range(COURTS) for t in range(SLOTS_PER_DAY))
        util = used_slots / (COURTS * SLOTS_PER_DAY * DAYS)

        # 2. 时间调整度 (Adj)
        adj = adj_score_total / max(self.R, 1)

        # 3. 机会公平性 (Fair): 基于满足率的差异（对应新公式6）
        club_sat = (club_slots / self.total_club_demand) if self.total_club_demand > 0 else 1.0
        free_sat = (free_slots / self.total_free_demand) if self.total_free_demand > 0 else 1.0
        fair = 1.0 - abs(club_sat - free_sat)

        # 4. 惩罚项 (Pen): 按完整规划周期总容量归一化（对应新公式7）
        total_capacity = DAYS * COURTS * SLOTS_PER_DAY
        pen_raw = unallocated_rigid_slots / total_capacity
        pen_norm = min(pen_raw, 1.0)

        # 综合评分
        score = (self.p.w_util * util + self.p.w_adj * adj + self.p.w_fair * fair - self.p.w_pen * pen_norm)

        indiv.fitness = score
        indiv.comp = {"util": util, "adj": adj, "fair": fair, "pen": pen_norm}
        return indiv


# ===================== 个体构造、修复与交叉 =====================

def empty_individual(R: int) -> Individual:
    return Individual([0] * R)


def initial_individual(reqs: List[Request]) -> Individual:
    """ 贪婪初始化：按优先级从高到低放置至最靠近偏好的可行位置。
    修正：HARD 仅允许放在“期望日+期望起始时段”（场馆若缺省则不限），
    不再退化到任意候选时间；若被 CLUB/FREE 阻塞则尝试嵌入式让位。
    """
    occ = Occupancy()
    genes = [0] * len(reqs)
    idxs = sorted(range(len(reqs)), key=lambda i: TYPE_ORDER.index(reqs[i].rtype))
    for i in idxs:
        r = reqs[i]
        cands = candidate_starts(r)
        cands = sort_candidates(r, cands, occ)
        placed = False
        if r.rtype == HARD:
            # 必须匹配期望日与起始时段；若无偏好场馆则场馆不限
            d, s = r.pref_day, r.pref_start
            courts = [r.pref_court] if r.pref_court is not None else list(range(COURTS))
            for c in courts:
                if occ.can_place(r, d, c, s):
                    occ.place(r, d, c, s)
                    genes[i] = enc(d, c, s)
                    placed = True
                    break
            if not placed:
                # 对每个场馆尝试让位
                for c in courts:
                    if try_embed_place_with_eviction(occ, reqs, genes, r, d, c, s):
                        placed = True
                        break
        else:
            for d, c, s in cands:
                if occ.can_place(r, d, c, s):
                    occ.place(r, d, c, s)
                    genes[i] = enc(d, c, s)
                    placed = True
                    break
            if (not placed) and (r.rtype in MUST_HAVE):
                for d, c, s in cands:
                    if d != r.pref_day:
                        continue
                    if try_embed_place_with_eviction(occ, reqs, genes, r, d, c, s):
                        placed = True
                        break
    return Individual(genes)


def random_feasible_individual(reqs: List[Request]) -> Individual:
    """ 随机化的可行个体构造：
    - 在类型优先级内引入随机打散；
    - 非 HARD 的候选起点随机尝试；
    - HARD 仍强制在 (pref_day, pref_start) 上，若需则尝试让位；
    """
    rng = random
    occ = Occupancy()
    genes = [0] * len(reqs)
    idxs = sorted(range(len(reqs)), key=lambda i: (TYPE_ORDER.index(reqs[i].rtype), rng.random()))
    for i in idxs:
        r = reqs[i]
        if r.rtype == HARD:
            d, s = r.pref_day, r.pref_start
            courts = [r.pref_court] if r.pref_court is not None else list(range(COURTS))
            rng.shuffle(courts)
            placed = False
            for c in courts:
                if occ.can_place(r, d, c, s):
                    occ.place(r, d, c, s)
                    genes[i] = enc(d, c, s)
                    placed = True
                    break
            if not placed:
                for c in courts:
                    if try_embed_place_with_eviction(occ, reqs, genes, r, d, c, s):
                        placed = True
                        break
            continue
        cands = candidate_starts(r)
        rng.shuffle(cands)
        placed = False
        for d, c, s in cands:
            if occ.can_place(r, d, c, s):
                occ.place(r, d, c, s)
                genes[i] = enc(d, c, s)
                placed = True
                break
        if (not placed) and (r.rtype in MUST_HAVE):
            # 尝试在偏好日嵌入
            pref_day_cands = [(r.pref_day, c, s) for (d, c, s) in cands if d == r.pref_day]
            for d, c, s in pref_day_cands:
                if try_embed_place_with_eviction(occ, reqs, genes, r, d, c, s):
                    break
    return Individual(genes)


def try_embed_place_with_eviction(occ: Occupancy, reqs: List[Request], genes: List[int], r: Request,
                                  d: int, c: int, s: int) -> bool:
    # 仅对 CLUB/FREE 进行嵌入式迁移
    blockers: List[Tuple[int, Request, Tuple[int, int, int]]] = []
    for i, gene in enumerate(genes):
        if gene == 0:
            continue
        q = reqs[i]
        if q.rtype not in (CLUB, FREE):
            continue
        qd, qc, qs = dec(gene)
        if qd == d and qc == c:
            if not (qs + q.duration <= s or s + r.duration <= qs):
                blockers.append((i, q, (qd, qc, qs)))
    temp_moves: List[Tuple[int, Tuple[int, int, int]]] = []
    for i, q, (qd, qc, qs) in blockers:
        repl = find_nearby_slot_for(occ, q, qd, qc, qs)
        if repl is None:
            return False
        temp_moves.append((i, repl))
    for i, (qd2, qc2, qs2) in temp_moves:
        q = reqs[i]
        od, oc, os = dec(genes[i])
        occ.remove(q, od, oc, os)
        occ.place(q, qd2, qc2, qs2)
        genes[i] = enc(qd2, qc2, qs2)
    if occ.can_place(r, d, c, s):
        occ.place(r, d, c, s)
        genes[r.rid] = enc(d, c, s)
        return True
    return False


def find_nearby_slot_for(occ: Occupancy, r: Request, d: int, c: int, s: int) -> Optional[Tuple[int, int, int]]:
    for radius in range(0, 6):
        for ds in [-radius, radius]:
            ns = s + ds
            if 0 <= ns <= SLOTS_PER_DAY - r.duration:
                if occ.can_place(r, d, c, ns):
                    return d, c, ns
        for court in range(COURTS):
            if court == c:
                continue
            for ds in [-radius, radius]:
                ns = s + ds
                if 0 <= ns <= SLOTS_PER_DAY - r.duration:
                    if occ.can_place(r, d, court, ns):
                        return d, court, ns
    for day_shift in [-1, 1]:
        d2 = d + day_shift
        if 0 <= d2 < DAYS:
            for ns in range(max(0, s - 1), min(SLOTS_PER_DAY - r.duration, s + 1) + 1):
                if occ.can_place(r, d2, c, ns):
                    return d2, c, ns
    return None


def crossover_constructive(reqs: List[Request], p1: Individual, p2: Individual) -> Individual:
    """逐请求构造式交叉（修正版）：
    - HARD 只允许在期望日+起始时段（场馆可不限）；失败仅尝试让位，不会退化到任意时间。
    """
    R = len(reqs)
    occ = Occupancy()
    child = [0] * R

    def single_adj_score(r: Request, gene: int) -> float:
        if gene == 0:
            return -1.0
        d, c, s = dec(gene)
        if d == r.pref_day:
            return 1.0 - 0.05 * abs(s - r.pref_start)
        else:
            return 0.9 - 0.25 * abs(d - r.pref_day) - 0.05 * abs(s - r.pref_start)

    idxs = sorted(range(R), key=lambda i: TYPE_ORDER.index(reqs[i].rtype))

    for i in idxs:
        r = reqs[i]
        g1, g2 = p1.genes[i], p2.genes[i]
        chosen = 0
        if g1 or g2:
            bestg = max([g for g in (g1, g2) if g], key=lambda g: single_adj_score(r, g), default=0)
            if bestg:
                d, c, s = dec(bestg)
                if occ.can_place(r, d, c, s):
                    occ.place(r, d, c, s)
                    child[i] = enc(d, c, s)
                    continue
                else:
                    chosen = bestg
        cands = candidate_starts(r)
        cands = sort_candidates(r, cands, occ)
        placed = False

        if r.rtype == HARD:
            d0, s0 = r.pref_day, r.pref_start
            courts = [r.pref_court] if r.pref_court is not None else list(range(COURTS))
            for c0 in courts:
                if occ.can_place(r, d0, c0, s0):
                    occ.place(r, d0, c0, s0)
                    child[i] = enc(d0, c0, s0)
                    placed = True
                    break
            if not placed:
                # 对每个场馆尝试让位
                for c0 in courts:
                    if try_embed_place_with_eviction(occ, reqs, child, r, d0, c0, s0):
                        placed = True
                        break
        else:
            if not placed:
                if chosen:
                    d, c, s = dec(chosen)
                    cands = [(d, c, s)] + [x for x in cands if x != (d, c, s)]
                for d, c, s in cands:
                    if occ.can_place(r, d, c, s):
                        occ.place(r, d, c, s)
                        child[i] = enc(d, c, s)
                        placed = True
                        break
            if (not placed) and (r.rtype in MUST_HAVE):
                for d, c, s in cands:
                    if d != r.pref_day:
                        continue
                    if try_embed_place_with_eviction(occ, reqs, child, r, d, c, s):
                        placed = True
                        break
        if (not placed) and (r.rtype in MUST_HAVE):
            return p1 if p1.fitness >= p2.fitness else p2
    return Individual(child)


# ===================== 变异 =====================

def mutate(reqs: List[Request], indiv: Individual, pm: float = 0.15) -> Individual:
    R = len(reqs)
    occ = Occupancy()
    for i, gene in enumerate(indiv.genes):
        if gene:
            d, c, s = dec(gene)
            occ.place(reqs[i], d, c, s)

    child = indiv.genes.copy()
    rng = random

    # 1) 同日微移
    for i in range(R):
        if rng.random() > pm:
            continue
        r = reqs[i]
        if r.rtype == HARD:
            continue
        old = child[i]
        if old == 0:
            continue
        d, c, s = dec(old)
        for delta in rng.sample([-2, -1, 1, 2], k=4):
            ns = s + delta
            if 0 <= ns <= SLOTS_PER_DAY - r.duration and occ.can_place(r, d, c, ns):
                occ.remove(r, d, c, s)
                occ.place(r, d, c, ns)
                child[i] = enc(d, c, ns)
                break

    # 2) 同日互换/对称滑移
    for _ in range(max(1, R // 30)):
        i, j = rng.sample(range(R), 2)
        ri, rj = reqs[i], reqs[j]
        gi, gj = child[i], child[j]
        if gi == 0 or gj == 0:
            continue
        di, ci, si = dec(gi)
        dj, cj, sj = dec(gj)
        if di != dj or ci != cj:
            continue
        if (ri.rtype != HARD) and (rj.rtype != HARD):
            occ.remove(ri, di, ci, si)
            occ.remove(rj, dj, cj, sj)
            ok_i = occ.can_place(ri, dj, cj, sj)
            ok_j = occ.can_place(rj, di, ci, si)
            if ok_i and ok_j:
                occ.place(ri, dj, cj, sj)
                occ.place(rj, di, ci, si)
                child[i], child[j] = enc(dj, cj, sj), enc(di, ci, si)
            else:
                occ.place(ri, di, ci, si)
                occ.place(rj, dj, cj, sj)

    # 3) 自由锻炼容量填充
    for i in range(R):
        r = reqs[i]
        if r.rtype != FREE:
            continue
        if child[i] != 0:
            continue
        cands = candidate_starts(r)
        rng.shuffle(cands)
        for d, c, s in cands:
            if occ.can_place(r, d, c, s):
                occ.place(r, d, c, s)
                child[i] = enc(d, c, s)
                break

    return Individual(child)


# ===================== 学习空间 =====================

def active_learning(reqs: List[Request], learner: Individual, teacher: Individual, fit: Fitness) -> Individual:
    if (learner.comp.get("adj", 0) + (1 - learner.comp.get("pen", 0))) < (
            teacher.comp.get("adj", 0) + (1 - teacher.comp.get("pen", 0))):
        high = {HARD, TEACHING, SOFT}
    else:
        high = {CLUB, FREE}

    occ = Occupancy()
    child = learner.genes.copy()
    for i, g in enumerate(child):
        if g:
            d, c, s = dec(g)
            occ.place(reqs[i], d, c, s)

    for i, r in enumerate(reqs):
        if r.rtype not in high:
            continue
        tg = teacher.genes[i]
        if tg == 0:
            continue
        d, c, s = dec(tg)
        if occ.can_place(r, d, c, s):
            if child[i]:
                od, oc, os = dec(child[i])
                occ.remove(r, od, oc, os)
            occ.place(r, d, c, s)
            child[i] = enc(d, c, s)
        else:
            cands = candidate_starts(r)
            cands.sort(key=lambda x: (abs(x[0] - d), abs(x[2] - s), 0 if (r.pref_court is None or x[1] == c) else 1))
            placed = False
            if child[i]:
                od, oc, os = dec(child[i])
                occ.remove(r, od, oc, os)
            for d2, c2, s2 in cands:
                if occ.can_place(r, d2, c2, s2):
                    occ.place(r, d2, c2, s2)
                    child[i] = enc(d2, c2, s2)
                    placed = True
                    break
            if not placed:
                if learner.genes[i]:
                    d0, c0, s0 = dec(learner.genes[i])
                    if occ.can_place(r, d0, c0, s0):
                        occ.place(r, d0, c0, s0)
                        child[i] = enc(d0, c0, s0)
                else:
                    child[i] = 0

    return Individual(child)


def passive_learning(reqs: List[Request], me: Individual, neighbor: Individual) -> Individual:
    high = {HARD, TEACHING, SOFT}
    block = high if random.random() < 0.5 else {CLUB, FREE}

    occ = Occupancy()
    child = me.genes.copy()
    for i, g in enumerate(child):
        if g:
            d, c, s = dec(g)
            occ.place(reqs[i], d, c, s)

    for i, r in enumerate(reqs):
        if r.rtype not in block:
            continue
        ng = neighbor.genes[i]
        if ng == 0:
            continue
        d, c, s = dec(ng)
        if occ.can_place(r, d, c, s):
            if child[i]:
                od, oc, os = dec(child[i])
                occ.remove(r, od, oc, os)
            occ.place(r, d, c, s)
            child[i] = enc(d, c, s)
    return Individual(child)


# ===================== 信仰空间（精英） =====================
class BeliefSpace:
    def __init__(self, k: int = 10):
        self.k = k
        self.pool: List[Individual] = []

    def accept(self, pop: List[Individual]):
        def key(ind: Individual):
            return (
                ind.comp.get("util", 0.0),
                ind.comp.get("adj", 0.0),
                ind.comp.get("fair", 0.0),
                -ind.comp.get("pen", 0.0),
                ind.fitness,
            )

        cand = sorted(pop, key=key, reverse=True)[: self.k]
        seen = set()
        unique = []
        for x in cand + self.pool:
            keyg = tuple(x.genes)
            if keyg in seen:
                continue
            seen.add(keyg)
            unique.append(x)
        self.pool = sorted(unique, key=lambda z: z.fitness, reverse=True)[: self.k]

    def influence(self, pop: List[Individual]) -> None:
        if not self.pool:
            return
        worst_idx = sorted(range(len(pop)), key=lambda i: pop[i].fitness)[: min(self.k, len(self.pool))]
        for j, elite in zip(worst_idx, self.pool):
            pop[j] = elite.clone()


# ===================== 文化交流空间（多区域） =====================
@dataclass
class Region:
    pop: List[Individual]
    belief: BeliefSpace
    strength: float = 0.0


def rank_exchange_rates(regions: List[Region]) -> Tuple[List[float], List[float]]:
    strengths = [max(1e-9, sum(x.fitness for x in r.belief.pool)) for r in regions]
    rank = sorted(range(len(regions)), key=lambda i: strengths[i], reverse=True)
    n = len(regions)
    recv_raw = [0.0] * n
    send_raw = [0.0] * n
    for pos, i in enumerate(rank):
        recv_raw[i] = n - pos
        send_raw[i] = pos + 1
    s1, s2 = sum(recv_raw), sum(send_raw)
    recv = [x / s1 for x in recv_raw]
    send = [x / s2 for x in send_raw]
    return send, recv


def individual_ranks(pop: List[Individual]) -> Tuple[List[float], List[float]]:
    n = len(pop)
    order = sorted(range(n), key=lambda i: pop[i].fitness, reverse=True)
    recv_raw = [0.0] * n
    send_raw = [0.0] * n
    for pos, i in enumerate(order):
        recv_raw[i] = n - pos
        send_raw[i] = pos + 1
    s1, s2 = sum(recv_raw), sum(send_raw)
    recv = [x / s1 for x in recv_raw]
    send = [x / s2 for x in send_raw]
    return send, recv


def cultural_exchange(regions: List[Region], reqs: List[Request], fit: Fitness, prob: float = 0.25):
    send_r, recv_r = rank_exchange_rates(regions)
    per_region_rates = []
    for reg in regions:
        send_i, recv_i = individual_ranks(reg.pop)
        per_region_rates.append((send_i, recv_i))
    for ri, reg in enumerate(regions):
        pop = reg.pop
        send_i, recv_i = per_region_rates[ri]
        for i, me in enumerate(pop):
            p_trigger = prob * send_r[ri] * send_i[i]
            if random.random() >= p_trigger:
                continue
            candidates = []
            weights = []
            for rj, other in enumerate(regions):
                if rj == ri:
                    continue
                recv_j = per_region_rates[rj][1]
                for k, ind in enumerate(other.pop):
                    candidates.append((rj, k))
                    weights.append(recv_r[rj] * recv_j[k])
            if not candidates:
                continue
            target_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
            rj, kj = candidates[target_idx]
            mate = regions[rj].pop[kj]
            child = crossover_constructive(reqs, me, mate)
            fit.eval(child)
            if child.fitness >= me.fitness:
                pop[i] = child


# ===================== 新增：预算对齐、退火、移民 工具函数 =====================

def effective_generations(pop_size: int, regions: int, fe_target: Optional[int]) -> Optional[int]:
    """基于 FE≈N0 + N*G（N0 初始评估）估算有效代数 G。返回 None 表示不改动。"""
    if not fe_target or fe_target <= 0:
        return None
    total_pop = pop_size * max(1, regions)
    return max(1, (fe_target - total_pop) // total_pop)


def annealed_probs(gargs: 'GAParams', gen: int, G: int) -> Tuple[float, float]:
    """按 burn-in + 幂退火 计算本代主动/被动学习概率。"""
    burn_in = int(gargs.burn_in_frac * G)
    if gen <= burn_in:
        return 0.0, 0.0
    t = (gen - burn_in) / max(1, G - burn_in)
    p_act = max(gargs.p_active_min, gargs.p_active0 * pow(1 - t, gargs.anneal_alpha))
    p_pas = max(gargs.p_passive_min, gargs.p_passive0 * pow(1 - t, gargs.anneal_beta))
    return p_act, p_pas


def diversity_hamming(pop: List[Individual]) -> float:
    """平均汉明距离（归一化到[0,1]）。"""
    N = len(pop)
    if N < 2:
        return 0.0
    L = len(pop[0].genes) if N else 0
    if L == 0:
        return 0.0
    s = 0.0
    cnt = 0
    for i in range(N):
        gi = pop[i].genes
        for j in range(i + 1, N):
            gj = pop[j].genes
            d = sum(1 for a, b in zip(gi, gj) if a != b) / L
            s += d
            cnt += 1
    return (2 * s) / (N * (N - 1)) if cnt else 0.0


def immigrate_if_needed(pop: List[Individual], fit: Fitness, reqs: List[Request],
                        D_min: float, rho: float, cooldown: int, elite_keep: int,
                        gen: int, last_imm_gen: int) -> int:
    if (gen - last_imm_gen) <= cooldown:
        return last_imm_gen
    D = diversity_hamming(pop)
    if D >= D_min:
        return last_imm_gen
    m = max(2, round(rho * len(pop)))
    immigrants = []
    for _ in range(m):
        ind = random_feasible_individual(reqs)
        fit.eval(ind)
        immigrants.append(ind)
    pop.sort(key=lambda z: z.fitness, reverse=True)
    # 保护前 elite_keep 个精英；用移民替换最后 m 个
    keep = max(0, min(elite_keep, len(pop) - m))
    pop[-m:] = immigrants
    return gen


# ===================== 主循环（CE_SLO 并行） =====================
@dataclass
class GAParams:
    pop_size: int = 100
    regions: int = 10
    gens: int = 800
    pc: float = 0.85
    pm: float = 0.2
    elite_k: int = 10
    influence_period: int = 10
    exchange_period: int = 5
    # 并行控制
    parallel: bool = True
    workers: Optional[int] = None
    # 学习退火参数（新增）
    burn_in_frac: float = 0.10  # 前10%代不学习
    p_active0: float = 0.60  # 主动学习初始概率（burn-in 后）
    p_passive0: float = 0.20  # 被动学习初始概率
    p_active_min: float = 0.10  # 主动学习下限
    p_passive_min: float = 0.05  # 被动学习下限
    anneal_alpha: float = 2.0  # 主动学习退火幂指数（更快衰减）
    anneal_beta: float = 1.0  # 被动学习退火幂指数
    # 预算对齐（新增）
    fe_target: Optional[int] = None  # 目标适应度评估次数；None 表示关闭
    # 随机移民（新增）
    diversity_min: float = 0.15  # 触发阈值（平均汉明距离）
    immigrants_rho: float = 0.05  # 每次注入比例（相对于种群规模）
    immigrants_cooldown: int = 5  # 触发冷却（代）
    immigrants_elite_keep: int = 2  # 保留的精英数
    use_fe_budget: bool = True  # True=按FE目标对齐；False=按固定gens运行

def roulette(pop: List[Individual]) -> Individual:
    s = sum(max(0.0, ind.fitness) for ind in pop)
    if s <= 0:
        return random.choice(pop)
    r = random.random() * s
    acc = 0.0
    for ind in pop:
        acc += max(0.0, ind.fitness)
        if acc >= r:
            return ind
    return pop[-1]


# ========= 子进程推进一代 =========

def evolve_region_one_gen(ri: int,
                          region_state: Region,
                          reqs: List[Request],
                          fit_params: FitnessParams,
                          gargs: GAParams,
                          gen: int,
                          seed: int) -> Tuple[int, Region, float]:
    """子进程里推进第 ri 个区域一代：微空间 +（小概率）学习 + accept"""
    # --- 【关键修复 1】同步全局变量，防止 Windows 子进程依旧使用 DAYS=7 导致数组越界崩溃 ---
    configure_horizon_from_requests(reqs)

    # 为可复现，给“区域×代”独立种子
    random.seed(seed + 7919 * ri + 104729 * gen)
    fit = Fitness(reqs, fit_params)

    # 预算对齐用有效代数，确保退火节奏一致
    if gargs.use_fe_budget and gargs.fe_target:
        G_eff = effective_generations(gargs.pop_size, gargs.regions, gargs.fe_target)
    else:
        G_eff = gargs.gens
    p_act, p_pas = annealed_probs(gargs, gen, G_eff)

    reg = region_state
    new_pop: List[Individual] = []
    elites = sorted(reg.pop, key=lambda z: z.fitness, reverse=True)[: max(1, gargs.pop_size // 10)]
    new_pop.extend(x.clone() for x in elites)

    while len(new_pop) < gargs.pop_size:
        p1 = roulette(reg.pop)
        if random.random() < gargs.pc:
            p2 = roulette(reg.pop)
            child = crossover_constructive(reqs, p1, p2)
        else:
            child = p1.clone()
        child = mutate(reqs, child, gargs.pm)

        # 学习（退火概率）
        if random.random() < p_act:
            teacher = elites[0]
            child = active_learning(reqs, child, teacher, fit)
        elif random.random() < p_pas:
            neighbor = roulette(reg.pop)
            child = passive_learning(reqs, child, neighbor)

        fit.eval(child)
        new_pop.append(child)

    reg.pop = new_pop
    reg.belief.accept(reg.pop)
    rbest = max(ind.fitness for ind in reg.pop)
    return ri, reg, rbest


def evolve_ce_slo_parallel(reqs: List[Request], fit: Fitness, gargs: GAParams, seed: int = 0,
                           log_csv_path: Optional[str] = None, verbose: bool = True, progress: bool = False) -> Tuple[
    Individual, List[Region]]:
    """区域级并行：每代子进程推进各区，代末统一 influence + 文化交流 + 移民"""
    random.seed(seed)

    # 初始化区域
    regions: List[Region] = []
    for _ in range(gargs.regions):
        pop = []
        for _ in range(gargs.pop_size):
            ind = initial_individual(reqs)
            fit.eval(ind)
            pop.append(ind)
        reg = Region(pop=pop, belief=BeliefSpace(gargs.elite_k))
        reg.belief.accept(reg.pop)
        regions.append(reg)

    best = max((ind for reg in regions for ind in reg.pop), key=lambda z: z.fitness)

    # 日志
    log_writer = None
    lf = None
    if log_csv_path is not None:
        lf = open(log_csv_path, "w", newline='', encoding='utf-8')
        header = ["gen"] + [f"region_{i}" for i in range(len(regions))] + ["overall"]
        log_writer = csv.writer(lf)
        log_writer.writerow(header)

    # --- Gen 000: 各区与整体最优（初始化，并行版）---
    region_bests = [max(ind.fitness for ind in reg.pop) for reg in regions]
    if verbose:
        reg_str = " ".join(f"R{i}={region_bests[i]:.4f}" for i in range(len(regions)))
        print(f"Gen 000 | {reg_str} | overall={best.fitness:.4f}")
    if log_writer is not None:
        row0 = [0] + [float(f"{x:.6f}") for x in region_bests] + [float(f"{best.fitness:.6f}")]
        log_writer.writerow(row0)

    # 预算对齐：有效代数
    if gargs.use_fe_budget and gargs.fe_target:
        G = effective_generations(gargs.pop_size, gargs.regions, gargs.fe_target)
    else:
        G = gargs.gens

    # 并行参数
    workers = gargs.workers or min(gargs.regions, os.cpu_count() or 1)
    fit_params = fit.p  # 仅传参即可在子进程重建 Fitness

    last_imm_gen = [-10 ** 9] * len(regions)

    if progress:
        print("[CE_SLO] Progress across generations:")
        _print_progress("CE_SLO gens", 0, G)

    # === timing setup ===
    MILESTONES = set(range(100, G + 1, 100))
    _t0 = time.perf_counter()  # 计时起点：仅统计算法演化阶段
    _hit = set()  # 已打印的里程碑
    _times = []  # 可选：收集到列表，最后汇总一行打印

    # --- 【关键修复 2】：将进程池移出循环！避免重复创建导致系统资源崩溃 ---
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for gen in range(1, G + 1):
            # 1) 区域并行推进一代（只 accept，不 influence）
            futures = [
                ex.submit(evolve_region_one_gen, ri, regions[ri], reqs, fit_params, gargs, gen, seed)
                for ri in range(len(regions))
            ]
            results = [f.result() for f in futures]

            # 2) 收集结果并按区域索引排序
            results.sort(key=lambda x: x[0])
            region_bests = []
            for ri, reg_state, rbest in results:
                regions[ri] = reg_state
                region_bests.append(rbest)

            # 3) 代末统一信仰影响
            if gargs.influence_period and gen % gargs.influence_period == 0:
                for reg in regions:
                    reg.belief.influence(reg.pop)
                    # 影响后确保适应度有效
                    for ind in reg.pop:
                        fit.eval(ind)

            # 4) 代末统一文化交流
            if gargs.exchange_period and gen % gargs.exchange_period == 0:
                cultural_exchange(regions, reqs, fit, prob=0.25)

            # 4.5) 区域内随机移民
            for ri, reg in enumerate(regions):
                last_imm_gen[ri] = immigrate_if_needed(
                    reg.pop, fit, reqs,
                    gargs.diversity_min, gargs.immigrants_rho,
                    gargs.immigrants_cooldown, gargs.immigrants_elite_keep,
                    gen, last_imm_gen[ri]
                )

            # 5) 统计与日志
            cur_best = max((ind for reg in regions for ind in reg.pop), key=lambda z: z.fitness)
            if cur_best.fitness > best.fitness:
                best = cur_best

            if verbose:
                reg_str = " ".join(f"R{i}={region_bests[i]:.4f}" for i in range(len(regions)))
                print(f"Gen {gen:03d} | {reg_str} | overall={best.fitness:.4f}")

            if progress:
                _print_progress("CE_SLO gens", gen, G)

            if log_writer is not None:
                row = [gen] + [float(f"{x:.6f}") for x in region_bests] + [float(f"{best.fitness:.6f}")]
                log_writer.writerow(row)

            if gen in MILESTONES and gen not in _hit:
                elapsed = time.perf_counter() - _t0
                print(f"[RUNTIME] iter={gen:>3d}  elapsed={elapsed:.3f}s", flush=True)
                _times.append((gen, elapsed))
                _hit.add(gen)

    if lf is not None:
        lf.close()

    if _times:
        summary = ", ".join(f"{it}:{t:.3f}s" for it, t in _times)
        print(f"[RUNTIME] summary -> {summary}")

    return best, regions


# ===================== 结果导出与可视化辅助 =====================

def save_best_schedule(path: str, reqs: List[Request], best: Individual):
    with open(path, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["rid", "type", "duration", "day", "court", "start"])
        for i, g in enumerate(best.genes):
            r = reqs[i]
            if g == 0:
                w.writerow([r.rid, r.rtype, r.duration, "UNASSIGNED", "", ""])
            else:
                d, c, s = dec(g)
                w.writerow([r.rid, r.rtype, r.duration, d, c, s])


def summarize(best: Individual):
    comp = best.comp
    return (
        f"Fitness={best.fitness:.4f} | util={comp.get('util', 0):.3f} "
        f"adj={comp.get('adj', 0):.3f} fair={comp.get('fair', 0):.3f} pen={comp.get('pen', 0):.3f}"
    )


# ===================== 单区域对比实验（Exp1/Exp2/Exp3） =====================

def active_learning_toward_teacher(reqs: List[Request], learner: Individual, teacher: Individual) -> Individual:
    """主动学习（传统 SLO 版本）：固定向当代最优个体学习。"""
    occ = Occupancy()
    child = learner.genes.copy()
    for i, g in enumerate(child):
        if g:
            d, c, s = dec(g)
            occ.place(reqs[i], d, c, s)

    idxs = sorted(range(len(reqs)), key=lambda i: TYPE_ORDER.index(reqs[i].rtype))
    for i in idxs:
        r = reqs[i]
        tg = teacher.genes[i]
        if tg == 0:
            continue
        d, c, s = dec(tg)
        # 先尝试完全跟随老师
        if occ.can_place(r, d, c, s):
            if child[i]:
                od, oc, os = dec(child[i])
                occ.remove(r, od, oc, os)
            occ.place(r, d, c, s)
            child[i] = enc(d, c, s)
            continue
        # 否则找最接近老师方案的可行位
        cands = candidate_starts(r)
        cands.sort(key=lambda x: (abs(x[0] - d), abs(x[2] - s), 0 if x[1] == c else 1))
        placed = False
        if child[i]:
            od, oc, os = dec(child[i])
            occ.remove(r, od, oc, os)
        for d2, c2, s2 in cands:
            if occ.can_place(r, d2, c2, s2):
                occ.place(r, d2, c2, s2)
                child[i] = enc(d2, c2, s2)
                placed = True
                break
        if not placed:
            # 失败则恢复原位（若有）
            if learner.genes[i]:
                d0, c0, s0 = dec(learner.genes[i])
                if occ.can_place(r, d0, c0, s0):
                    occ.place(r, d0, c0, s0)
                    child[i] = enc(d0, c0, s0)
            else:
                child[i] = 0
    return Individual(child)


def evolve_single_region(reqs: List[Request], fit: Fitness, gargs: GAParams, seed: int = 0,
                         mode: str = "GA", log_csv_path: Optional[str] = None, verbose: bool = True,
                         progress: bool = False) -> Tuple[Individual, List[Individual]]:
    """单区域对比实验主循环。"""
    random.seed(seed)
    pop: List[Individual] = []
    for _ in range(gargs.pop_size):
        ind = initial_individual(reqs)
        fit.eval(ind)
        pop.append(ind)

    belief = BeliefSpace(gargs.elite_k)
    belief.accept(pop)

    # 日志
    log_writer = None
    lf = None
    if log_csv_path is not None:
        lf = open(log_csv_path, "w", newline='', encoding='utf-8')
        log_writer = csv.writer(lf)
        log_writer.writerow(["gen", "region_0", "overall"])  # 单区域，region_0==overall

    best = max(pop, key=lambda z: z.fitness)
    per_gen_best: List[Individual] = []

    # --- Gen 000: 初始种群的最优 ---
    region_best = best  # 单区，region_0 == overall
    per_gen_best.append(best.clone())
    if verbose:
        print(f"Gen 000 | R0={region_best.fitness:.4f} | overall={best.fitness:.4f}")
    if log_writer is not None:
        log_writer.writerow([0, f"{region_best.fitness:.6f}", f"{best.fitness:.6f}"])

    # 预算对齐：确定有效代数
    if gargs.use_fe_budget and gargs.fe_target:
        G = effective_generations(gargs.pop_size, 1, gargs.fe_target)
    else:
        G = gargs.gens

    last_imm_gen = -10 ** 9

    if progress:
        print(f"[{mode}] Progress across generations:")
        _print_progress(f"{mode:>5} gens", 0, G)

    for gen in range(1, G + 1):
        # 本代退火后的学习概率
        p_act, p_pas = annealed_probs(gargs, gen, G)

        # 选拔精英
        elites = sorted(pop, key=lambda z: z.fitness, reverse=True)[: max(1, gargs.pop_size // 10)]
        new_pop: List[Individual] = [x.clone() for x in elites]

        # 产生后代
        while len(new_pop) < gargs.pop_size:
            p1 = roulette(pop)
            if random.random() < gargs.pc:
                p2 = roulette(pop)
                child = crossover_constructive(reqs, p1, p2)
            else:
                child = p1.clone()
            child = mutate(reqs, child, gargs.pm)

            # 学习空间（按模式 + 退火概率）
            if mode == "SLO":
                if random.random() < p_act:
                    teacher = elites[0]
                    child = active_learning_toward_teacher(reqs, child, teacher)
                if random.random() < p_pas:
                    neighbor = roulette(pop)
                    child = passive_learning(reqs, child, neighbor)

            fit.eval(child)
            new_pop.append(child)

        pop = new_pop

        # 信仰空间影响（按模式）
        if mode in ("CGA", "SLO") and (gargs.influence_period and gen % gargs.influence_period == 0):
            belief.accept(pop)
            belief.influence(pop)
            for ind in pop:
                fit.eval(ind)

        # 随机移民（多样性维持）
        last_imm_gen = immigrate_if_needed(
            pop, fit, reqs,
            gargs.diversity_min, gargs.immigrants_rho,
            gargs.immigrants_cooldown, gargs.immigrants_elite_keep,
            gen, last_imm_gen
        )

        # 记录 & 打印
        region_best = max(pop, key=lambda z: z.fitness)
        if region_best.fitness > best.fitness:
            best = region_best
        per_gen_best.append(best.clone())

        if verbose:
            print(f"Gen {gen:03d} | R0={region_best.fitness:.4f} | overall={best.fitness:.4f}")
        if progress:
            _print_progress(f"{mode:>5} gens", gen, G)
        if log_writer is not None:
            log_writer.writerow([gen, f"{region_best.fitness:.6f}", f"{best.fitness:.6f}"])

    if lf is not None:
        lf.close()

    return best, per_gen_best

def configure_horizon_from_requests(reqs: List[Request]) -> None:
    """根据数据自动设置 DAYS、TOTAL_SLOTS 与各类型可排日（跨周用 d%7 识别星期）"""
    global DAYS, TOTAL_SLOTS, ALLOWED_DAYS
    max_day = max(r.pref_day for r in reqs) if reqs else 6
    DAYS = max_day + 1
    TOTAL_SLOTS = DAYS * COURTS * SLOTS_PER_DAY

    # 0=Mon ... 6=Sun
    ALLOWED_DAYS = {
        HARD:      list(range(DAYS)),                              # 周一~周日
        SOFT:      list(range(DAYS)),                              # 周一~周日
        TEACHING:  [d for d in range(DAYS) if (d % 7) <= 4],       # 周一~周五
        CLUB:      list(range(DAYS)),                              # 周一~周日
        FREE:      [d for d in range(DAYS) if (d % 7) >= 5],       # 周六~周日
    }

# ===================== main =====================

def main(csv_path: Optional[str] = None, seed: int = 1997):
    # 1) 读入或生成请求
    if csv_path:
        try:
            reqs = load_requests_from_csv(csv_path)
            print(f"Loaded {len(reqs)} requests from {csv_path}")
        except Exception as e:
            print(f"[WARN] Failed to load CSV ({e}). Fallback to demo generator.")
            reqs = make_demo_requests(seed=seed)
    else:
        reqs = make_demo_requests(seed=seed)
        print(f"Generated demo requests: {len(reqs)}")

    # >>> 新增：根据数据扩展天数/允许日
    configure_horizon_from_requests(reqs)

    # 2) 构建适应度
    fit = Fitness(reqs, FitnessParams())

    # 3) 进化（多区域 CE_SLO）
    params = GAParams(parallel=True, use_fe_budget=False, gens=800)  # 预算对齐默认启用；可调整/置 None 关闭
    best, regions = evolve_ce_slo_parallel(reqs, fit, params, seed=seed,
                                           log_csv_path="all_evolve_log.csv_version_1.2.csv",
                                           verbose=False)

    print("Best:", summarize(best))

    # 4) 导出
    out = "all_best_schedule.csv_version_1.2.csv"
    save_best_schedule(out, reqs, best)
    print(f"Saved best schedule to {out}")


# =============== 对比实验入口（Exp1/2/3） ===============

def main_exp1(csv_path: Optional[str] = None, seed: int = 1997):
    """Exp1: 单区域，仅微空间（GA）。写日志到 exp1_micro_only.csv"""
    if csv_path:
        reqs = load_requests_from_csv(csv_path)
    else:
        reqs = make_demo_requests(seed=seed)
    fit = Fitness(reqs, FitnessParams())
    best, per_gen = evolve_single_region(
        reqs, fit, GAParams(parallel=False, fe_target=100_000), seed=seed,
        mode="GA", log_csv_path="exp1_micro_only_version_1.2.csv", verbose=False)
    print("[Exp1] Best:", summarize(best))
    save_best_schedule("exp1_best_schedule_version_1.2.csv", reqs, best)


def main_exp2(csv_path: Optional[str] = None, seed: int = 1997):
    """Exp2: 单区域，微空间+信仰空间（CGA）。写日志到 exp2_micro_belief.csv"""
    if csv_path:
        reqs = load_requests_from_csv(csv_path)
    else:
        reqs = make_demo_requests(seed=seed)
    fit = Fitness(reqs, FitnessParams())
    best, per_gen = evolve_single_region(
        reqs, fit, GAParams(parallel=False, fe_target=100_000), seed=seed,
        mode="CGA", log_csv_path="exp2_micro_belief_version_1.2.csv", verbose=False)
    print("[Exp2] Best:", summarize(best))
    save_best_schedule("exp2_best_schedule_version_1.2.csv", reqs, best)


def main_exp3(csv_path: Optional[str] = None, seed: int = 1997):
    """Exp3: 单区域，微空间+学习空间(主动=向最优学习)+信仰空间（SLO）。写日志到 exp3_micro_learning_belief.csv"""
    if csv_path:
        reqs = load_requests_from_csv(csv_path)
    else:
        reqs = make_demo_requests(seed=seed)
    fit = Fitness(reqs, FitnessParams())
    best, per_gen = evolve_single_region(
        reqs, fit, GAParams(parallel=False, fe_target=100_000), seed=seed,
        mode="SLO", log_csv_path="exp3_micro_learning_belief_version_1.2.csv", verbose=False)
    print("[Exp3] Best:", summarize(best))
    save_best_schedule("exp3_best_schedule_version_1.2.csv", reqs, best)


def main_all(csv_path: Optional[str] = None, seed: int = 1997):
    """依次执行 GA、CGA、SLO、CE_SLO（并行），全部完成后统一打印各自最优个体摘要，并各自落盘日志与最优调度。"""
    # 1) 加载请求
    if csv_path:
        reqs = load_requests_from_csv(csv_path)
    else:
        reqs = make_demo_requests(seed=seed)
    fit = Fitness(reqs, FitnessParams())

    # 2) 单区三种算法（统一FE预算）

    # 以“统一评价预算”执行代码
    # args_single = GAParams(parallel=False, use_fe_budget=True, fe_target=100_000)
    # 以设置“最大迭代次数”执行代码
    args_single = GAParams(parallel=False, use_fe_budget=False, gens=500)

    best_ga, _ = evolve_single_region(
        reqs, fit, args_single, seed=seed,
        mode="GA", log_csv_path="exp1_micro_only_version_1.2.csv", verbose=False, progress=True)
    save_best_schedule("exp1_best_schedule_version_1.2.csv", reqs, best_ga)

    best_cga, _ = evolve_single_region(
        reqs, fit, args_single, seed=seed,
        mode="CGA", log_csv_path="exp2_micro_belief_version_1.2.csv", verbose=False, progress=True)
    save_best_schedule("exp2_best_schedule_version_1.2.csv", reqs, best_cga)

    best_slo, _ = evolve_single_region(
        reqs, fit, args_single, seed=seed,
        mode="SLO", log_csv_path="exp3_micro_learning_belief_version_1.2.csv", verbose=False, progress=True)
    save_best_schedule("exp3_best_schedule_version_1.2.csv", reqs, best_slo)

    # 3) CE_SLO 并行（同样统一FE预算）

    # 以“统一评价预算”执行代码
    # args_ce = GAParams(parallel=True, regions=10, use_fe_budget=True, fe_target=100_000)
    # 以设置“最大迭代次数”执行代码
    args_ce = GAParams(parallel=True, regions=10, use_fe_budget=False, gens=500)

    best_ce, _regions = evolve_ce_slo_parallel(
        reqs, fit, args_ce, seed=seed,
        log_csv_path="exp4_ce_slo_version_1.2.csv", verbose=False, progress=True)
    save_best_schedule("exp4_best_schedule_version_1.2.csv", reqs, best_ce)

    # 4) 统一输出（全部结束后再打印）
    print("Summary after all finished:")
    print(f"GA     best: {summarize(best_ga)}")
    print(f"CGA    best: {summarize(best_cga)}")
    print(f"SLO    best: {summarize(best_slo)}")
    print(f"CE_SLO best: {summarize(best_ce)}")


if __name__ == "__main__":
    # 随机种子数命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1997, help="Random seed")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV (research or legacy)")
    args = parser.parse_args()

    main(csv_path=args.csv, seed=args.seed)
