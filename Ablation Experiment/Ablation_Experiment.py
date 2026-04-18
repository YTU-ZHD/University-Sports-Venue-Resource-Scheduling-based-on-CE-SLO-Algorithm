from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import argparse
import dataclasses
import time

def _print_progress(prefix: str, i: int, total: int, width: int = 30):
    total = max(1, int(total))
    i = min(max(0, int(i)), total)
    filled = int(width * i / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix} [{bar}] {i}/{total}", end="", flush=True)
    if i >= total:
        print()

DAYS = 7
COURTS = 3
SLOTS_PER_DAY = 12
TOTAL_SLOTS = DAYS * COURTS * SLOTS_PER_DAY  # 252

CAPACITY_NONFREE = 1
CAPACITY_FREE = 10

HARD, TEACHING, SOFT, CLUB, FREE = "HARD", "TEACHING", "SOFT", "CLUB", "FREE"
TYPE_ORDER = [HARD, TEACHING, SOFT, CLUB, FREE]
MUST_HAVE = {HARD, TEACHING, SOFT}

ALLOWED_DAYS = {
    HARD: list(range(0, 7)),
    SOFT: list(range(0, 7)),
    TEACHING: list(range(0, 5)),
    CLUB: list(range(0, 7)),
    FREE: [5, 6],
}

DAY_NAME_TO_IDX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6
}


def enc(day: int, court: int, start: int) -> int:
    return day * (COURTS * SLOTS_PER_DAY) + court * SLOTS_PER_DAY + start + 1


def dec(code: int) -> Tuple[int, int, int]:
    idx = code - 1
    day = idx // (COURTS * SLOTS_PER_DAY)
    rem = idx % (COURTS * SLOTS_PER_DAY)
    court = rem // SLOTS_PER_DAY
    start = rem % SLOTS_PER_DAY
    return day, court, start

@dataclass
class Request:
    rid: int
    rtype: str
    duration: int
    pref_day: int
    pref_start: int
    pref_court: Optional[int]

    def __hash__(self):
        return hash(self.rid)

@dataclass
class Individual:
    genes: List[int]
    fitness: float = 0.0
    comp: Dict[str, float] = field(default_factory=dict)

    def clone(self) -> 'Individual':
        return Individual(self.genes.copy(), self.fitness, self.comp.copy())

def make_demo_requests(seed: int = 42) -> List[Request]:
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
            reqs.append(Request(len(reqs), rtype, dur, d, start, court))

    add(8, HARD, [3, 4])
    add(12, SOFT, [3, 4])
    add(60, TEACHING, [2])
    add(40, CLUB, [1, 2, 3])
    add(50, FREE, [1])
    return reqs

def _coerce_day(x) -> int:
    if isinstance(x, int): return x
    s = str(x).strip().lower()
    if s in DAY_NAME_TO_IDX: return DAY_NAME_TO_IDX[s]
    raise ValueError(f"Unrecognized day: {x}")

def load_requests_from_csv(path: str) -> List[Request]:
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        idx = {h.lower(): i for i, h in enumerate(header)}
        reqs: List[Request] = []
        if {"requestid", "type", "desiredday", "desiredslot", "duration"} <= set(idx.keys()):
            tmap = {"hard game": HARD, "soft game": SOFT, "teaching": TEACHING, "student club": CLUB,
                    "free exercise": FREE}
            for row in reader:
                rtype = tmap[row[idx["type"]].strip().lower()]
                d = _coerce_day(row[idx["desiredday"]])
                s = int(row[idx["desiredslot"]]) - 1
                dur = int(row[idx["duration"]])
                reqs.append(Request(len(reqs), rtype, dur, d, s, None))
            return reqs
        needed = ["id", "type", "preferred_day", "preferred_start", "duration"]
        for key in needed:
            if key not in idx: raise ValueError(f"CSV missing column: {key}")
        has_pc = ("preferred_court" in idx)
        for row in reader:
            rtype = row[idx["type"]].strip().upper()
            pd = int(row[idx["preferred_day"]])
            ps = int(row[idx["preferred_start"]])
            dur = int(row[idx["duration"]])
            pc = int(row[idx["preferred_court"]].strip()) if has_pc and row[
                idx["preferred_court"]].strip() != "" else None
            reqs.append(Request(len(reqs), rtype, dur, pd, ps, pc))
        return reqs

class Occupancy:
    def __init__(self):
        self.nonfree = [[[0 for _ in range(SLOTS_PER_DAY)] for _ in range(COURTS)] for _ in range(DAYS)]
        self.free = [[[0 for _ in range(SLOTS_PER_DAY)] for _ in range(COURTS)] for _ in range(DAYS)]

    def can_place(self, r: Request, day: int, court: int, start: int) -> bool:
        if day not in ALLOWED_DAYS[r.rtype] or start + r.duration > SLOTS_PER_DAY: return False
        if r.rtype == FREE:
            for t in range(start, start + r.duration):
                if self.nonfree[day][court][t] > 0 or self.free[day][court][t] >= CAPACITY_FREE: return False
            return True
        else:
            for t in range(start, start + r.duration):
                if self.nonfree[day][court][t] > 0 or self.free[day][court][t] > 0: return False
            return True

    def place(self, r: Request, day: int, court: int, start: int):
        if r.rtype == FREE:
            for t in range(start, start + r.duration): self.free[day][court][t] += 1
        else:
            for t in range(start, start + r.duration): self.nonfree[day][court][t] += 1

    def remove(self, r: Request, day: int, court: int, start: int):
        if r.rtype == FREE:
            for t in range(start, start + r.duration): self.free[day][court][t] -= 1
        else:
            for t in range(start, start + r.duration): self.nonfree[day][court][t] -= 1

def candidate_starts(r: Request) -> List[Tuple[int, int, int]]:
    cands = []
    for d in ALLOWED_DAYS[r.rtype]:
        for court in range(COURTS):
            for s in range(0, SLOTS_PER_DAY - r.duration + 1):
                cands.append((d, court, s))
    return cands

def sort_candidates(r: Request, cands: List[Tuple[int, int, int]], occ: Occupancy) -> List[Tuple[int, int, int]]:
    def key(c):
        d, court, s = c
        return (abs(d - r.pref_day), abs(s - r.pref_start), 0 if (r.pref_court is None or court == r.pref_court) else 1)

    return sorted(cands, key=key)

@dataclass
class FitnessParams:
    w_util: float = 0.35
    w_adj: float = 0.35
    w_fair: float = 0.2
    w_pen: float = 0.1
    adj_same_max: float = 1.0
    adj_cross_day_max: float = 0.9
    day_alpha: float = 0.25
    slot_beta: float = 0.05

class Fitness:
    def __init__(self, reqs: List[Request], params: FitnessParams):
        self.reqs = reqs
        self.R = len(reqs)
        self.p = params
        self.total_club_demand = sum(r.duration for r in reqs if r.rtype == CLUB)
        self.total_free_demand = sum(r.duration for r in reqs if r.rtype == FREE)

    def eval(self, indiv: Individual) -> Individual:
        adj_score_total = 0.0
        unallocated_rigid_slots = 0.0
        club_slots = 0
        free_slots = 0
        used = [[[0 for _ in range(SLOTS_PER_DAY)] for _ in range(COURTS)] for _ in range(DAYS)]

        for i, gene in enumerate(indiv.genes):
            r = self.reqs[i]
            if gene == 0:
                if r.rtype in MUST_HAVE:
                    unallocated_rigid_slots += r.duration
                continue
            d, c, s = dec(gene)
            for t in range(s, s + r.duration):
                used[d][c][t] = 1

            if d == r.pref_day:
                adj_score_total += max(self.p.adj_same_max - self.p.slot_beta * abs(s - r.pref_start), 0.0)
            else:
                adj_score_total += max(
                    self.p.adj_cross_day_max - self.p.day_alpha * abs(d - r.pref_day) - self.p.slot_beta * abs(
                        s - r.pref_start), 0.0)

            if r.rtype == CLUB:
                club_slots += r.duration
            elif r.rtype == FREE:
                free_slots += r.duration

        used_slots = sum(used[d][c][t] for d in range(DAYS) for c in range(COURTS) for t in range(SLOTS_PER_DAY))
        util = used_slots / (COURTS * SLOTS_PER_DAY * DAYS)
        adj = adj_score_total / max(self.R, 1)

        club_sat = (club_slots / self.total_club_demand) if self.total_club_demand > 0 else 1.0
        free_sat = (free_slots / self.total_free_demand) if self.total_free_demand > 0 else 1.0
        fair = 1.0 - abs(club_sat - free_sat)

        total_capacity = DAYS * COURTS * SLOTS_PER_DAY
        pen_norm = min(unallocated_rigid_slots / total_capacity, 1.0)

        score = (self.p.w_util * util + self.p.w_adj * adj + self.p.w_fair * fair - self.p.w_pen * pen_norm)
        indiv.fitness = score
        indiv.comp = {"util": util, "adj": adj, "fair": fair, "pen": pen_norm}
        return indiv

def try_embed_place_with_eviction(occ: Occupancy, reqs: List[Request], genes: List[int], r: Request, d: int, c: int,
                                  s: int) -> bool:
    blockers = []
    for i, gene in enumerate(genes):
        if gene == 0: continue
        q = reqs[i]
        if q.rtype not in (CLUB, FREE): continue
        qd, qc, qs = dec(gene)
        if qd == d and qc == c and not (qs + q.duration <= s or s + r.duration <= qs):
            blockers.append((i, q, (qd, qc, qs)))
    temp_moves = []
    for i, q, (qd, qc, qs) in blockers:
        repl = find_nearby_slot_for(occ, q, qd, qc, qs)
        if repl is None: return False
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
            if 0 <= ns <= SLOTS_PER_DAY - r.duration and occ.can_place(r, d, c, ns): return d, c, ns
        for court in range(COURTS):
            if court == c: continue
            for ds in [-radius, radius]:
                ns = s + ds
                if 0 <= ns <= SLOTS_PER_DAY - r.duration and occ.can_place(r, d, court, ns): return d, court, ns
    for day_shift in [-1, 1]:
        d2 = d + day_shift
        if 0 <= d2 < DAYS:
            for ns in range(max(0, s - 1), min(SLOTS_PER_DAY - r.duration, s + 1) + 1):
                if occ.can_place(r, d2, c, ns): return d2, c, ns
    return None

def initial_individual(reqs: List[Request]) -> Individual:
    occ = Occupancy()
    genes = [0] * len(reqs)
    idxs = sorted(range(len(reqs)), key=lambda i: TYPE_ORDER.index(reqs[i].rtype))
    for i in idxs:
        r = reqs[i]
        cands = sort_candidates(r, candidate_starts(r), occ)
        placed = False
        if r.rtype == HARD:
            d, s = r.pref_day, r.pref_start
            courts = [r.pref_court] if r.pref_court is not None else list(range(COURTS))
            for c in courts:
                if occ.can_place(r, d, c, s):
                    occ.place(r, d, c, s);
                    genes[i] = enc(d, c, s);
                    placed = True;
                    break
            if not placed:
                for c in courts:
                    if try_embed_place_with_eviction(occ, reqs, genes, r, d, c, s):
                        placed = True;
                        break
        else:
            for d, c, s in cands:
                if occ.can_place(r, d, c, s):
                    occ.place(r, d, c, s);
                    genes[i] = enc(d, c, s);
                    placed = True;
                    break
            if (not placed) and (r.rtype in MUST_HAVE):
                for d, c, s in cands:
                    if d == r.pref_day and try_embed_place_with_eviction(occ, reqs, genes, r, d, c, s):
                        placed = True;
                        break
    return Individual(genes)

def random_feasible_individual(reqs: List[Request]) -> Individual:
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
                    occ.place(r, d, c, s);
                    genes[i] = enc(d, c, s);
                    placed = True;
                    break
            if not placed:
                for c in courts:
                    if try_embed_place_with_eviction(occ, reqs, genes, r, d, c, s):
                        placed = True;
                        break
            continue
        cands = candidate_starts(r)
        rng.shuffle(cands)
        placed = False
        for d, c, s in cands:
            if occ.can_place(r, d, c, s):
                occ.place(r, d, c, s);
                genes[i] = enc(d, c, s);
                placed = True;
                break
        if (not placed) and (r.rtype in MUST_HAVE):
            for d, c, s in [(d, c, s) for (d, c, s) in cands if d == r.pref_day]:
                if try_embed_place_with_eviction(occ, reqs, genes, r, d, c, s): break
    return Individual(genes)

def crossover_constructive(reqs: List[Request], p1: Individual, p2: Individual) -> Individual:
    R = len(reqs)
    occ = Occupancy()
    child = [0] * R

    def single_adj_score(r: Request, gene: int) -> float:
        if gene == 0: return -1.0
        d, c, s = dec(gene)
        return 1.0 - 0.05 * abs(s - r.pref_start) if d == r.pref_day else 0.9 - 0.25 * abs(d - r.pref_day) - 0.05 * abs(
            s - r.pref_start)

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
                    occ.place(r, d, c, s);
                    child[i] = enc(d, c, s);
                    continue
                else:
                    chosen = bestg
        cands = sort_candidates(r, candidate_starts(r), occ)
        placed = False
        if r.rtype == HARD:
            d0, s0 = r.pref_day, r.pref_start
            courts = [r.pref_court] if r.pref_court is not None else list(range(COURTS))
            for c0 in courts:
                if occ.can_place(r, d0, c0, s0):
                    occ.place(r, d0, c0, s0);
                    child[i] = enc(d0, c0, s0);
                    placed = True;
                    break
            if not placed:
                for c0 in courts:
                    if try_embed_place_with_eviction(occ, reqs, child, r, d0, c0, s0):
                        placed = True;
                        break
        else:
            if not placed:
                if chosen:
                    d, c, s = dec(chosen)
                    cands = [(d, c, s)] + [x for x in cands if x != (d, c, s)]
                for d, c, s in cands:
                    if occ.can_place(r, d, c, s):
                        occ.place(r, d, c, s);
                        child[i] = enc(d, c, s);
                        placed = True;
                        break
            if (not placed) and (r.rtype in MUST_HAVE):
                for d, c, s in cands:
                    if d == r.pref_day and try_embed_place_with_eviction(occ, reqs, child, r, d, c, s):
                        placed = True;
                        break
        if (not placed) and (r.rtype in MUST_HAVE): return p1 if p1.fitness >= p2.fitness else p2
    return Individual(child)

def mutate(reqs: List[Request], indiv: Individual, pm: float = 0.15) -> Individual:
    R = len(reqs)
    occ = Occupancy()
    for i, gene in enumerate(indiv.genes):
        if gene:
            d, c, s = dec(gene)
            occ.place(reqs[i], d, c, s)
    child = indiv.genes.copy()
    rng = random
    for i in range(R):
        if rng.random() > pm or reqs[i].rtype == HARD or child[i] == 0: continue
        r, (d, c, s) = reqs[i], dec(child[i])
        for delta in rng.sample([-2, -1, 1, 2], k=4):
            ns = s + delta
            if 0 <= ns <= SLOTS_PER_DAY - r.duration and occ.can_place(r, d, c, ns):
                occ.remove(r, d, c, s);
                occ.place(r, d, c, ns);
                child[i] = enc(d, c, ns);
                break
    for _ in range(max(1, R // 30)):
        i, j = rng.sample(range(R), 2)
        if child[i] == 0 or child[j] == 0 or reqs[i].rtype == HARD or reqs[j].rtype == HARD: continue
        di, ci, si = dec(child[i]);
        dj, cj, sj = dec(child[j])
        if di != dj or ci != cj: continue
        occ.remove(reqs[i], di, ci, si);
        occ.remove(reqs[j], dj, cj, sj)
        if occ.can_place(reqs[i], dj, cj, sj) and occ.can_place(reqs[j], di, ci, si):
            occ.place(reqs[i], dj, cj, sj);
            occ.place(reqs[j], di, ci, si)
            child[i], child[j] = enc(dj, cj, sj), enc(di, ci, si)
        else:
            occ.place(reqs[i], di, ci, si);
            occ.place(reqs[j], dj, cj, sj)
    for i in range(R):
        if reqs[i].rtype == FREE and child[i] == 0:
            cands = candidate_starts(reqs[i]);
            rng.shuffle(cands)
            for d, c, s in cands:
                if occ.can_place(reqs[i], d, c, s):
                    occ.place(reqs[i], d, c, s);
                    child[i] = enc(d, c, s);
                    break
    return Individual(child)

def active_learning(reqs: List[Request], learner: Individual, teacher: Individual, fit: Fitness) -> Individual:
    high = {HARD, TEACHING, SOFT} if (learner.comp.get("adj", 0) + (1 - learner.comp.get("pen", 0))) < (
                teacher.comp.get("adj", 0) + (1 - teacher.comp.get("pen", 0))) else {CLUB, FREE}
    occ = Occupancy()
    child = learner.genes.copy()
    for i, g in enumerate(child):
        if g:
            d, c, s = dec(g)
            occ.place(reqs[i], d, c, s)
    for i, r in enumerate(reqs):
        if r.rtype not in high or teacher.genes[i] == 0: continue
        d, c, s = dec(teacher.genes[i])
        if occ.can_place(r, d, c, s):
            if child[i]:
                od, oc, os = dec(child[i])
                occ.remove(r, od, oc, os)
            occ.place(r, d, c, s);
            child[i] = enc(d, c, s)
        else:
            cands = sort_candidates(r, candidate_starts(r), occ)
            placed = False
            if child[i]:
                od, oc, os = dec(child[i])
                occ.remove(r, od, oc, os)
            for d2, c2, s2 in cands:
                if occ.can_place(r, d2, c2, s2):
                    occ.place(r, d2, c2, s2);
                    child[i] = enc(d2, c2, s2);
                    placed = True;
                    break
            if not placed:
                if learner.genes[i]:
                    d0, c0, s0 = dec(learner.genes[i])
                    if occ.can_place(r, d0, c0, s0):
                        occ.place(r, d0, c0, s0);
                        child[i] = enc(d0, c0, s0)
                else:
                    child[i] = 0
    return Individual(child)

def passive_learning(reqs: List[Request], me: Individual, neighbor: Individual) -> Individual:
    block = {HARD, TEACHING, SOFT} if random.random() < 0.5 else {CLUB, FREE}
    occ = Occupancy()
    child = me.genes.copy()
    for i, g in enumerate(child):
        if g:
            d, c, s = dec(g)
            occ.place(reqs[i], d, c, s)
    for i, r in enumerate(reqs):
        if r.rtype not in block or neighbor.genes[i] == 0: continue
        d, c, s = dec(neighbor.genes[i])
        if occ.can_place(r, d, c, s):
            if child[i]:
                od, oc, os = dec(child[i])
                occ.remove(r, od, oc, os)
            occ.place(r, d, c, s);
            child[i] = enc(d, c, s)
    return Individual(child)

class BeliefSpace:
    def __init__(self, k: int = 10):
        self.k = k
        self.pool: List[Individual] = []

    def accept(self, pop: List[Individual]):
        cand = sorted(pop, key=lambda ind: (
        ind.comp.get("util", 0.0), ind.comp.get("adj", 0.0), ind.comp.get("fair", 0.0), -ind.comp.get("pen", 0.0),
        ind.fitness), reverse=True)[: self.k]
        seen, unique = set(), []
        for x in cand + self.pool:
            keyg = tuple(x.genes)
            if keyg not in seen:
                seen.add(keyg);
                unique.append(x)
        self.pool = sorted(unique, key=lambda z: z.fitness, reverse=True)[: self.k]

    def influence(self, pop: List[Individual]) -> None:
        if not self.pool: return
        worst_idx = sorted(range(len(pop)), key=lambda i: pop[i].fitness)[: min(self.k, len(self.pool))]
        for j, elite in zip(worst_idx, self.pool): pop[j] = elite.clone()

@dataclass
class Region:
    pop: List[Individual]
    belief: BeliefSpace

def rank_exchange_rates(regions: List[Region]) -> Tuple[List[float], List[float]]:
    strengths = [max(1e-9, sum(x.fitness for x in r.belief.pool)) for r in regions]
    rank = sorted(range(len(regions)), key=lambda i: strengths[i], reverse=True)
    n = len(regions)
    recv_raw, send_raw = [0.0] * n, [0.0] * n
    for pos, i in enumerate(rank):
        recv_raw[i] = n - pos
        send_raw[i] = pos + 1
    return [x / sum(send_raw) for x in send_raw], [x / sum(recv_raw) for x in recv_raw]

def individual_ranks(pop: List[Individual]) -> Tuple[List[float], List[float]]:
    n = len(pop)
    order = sorted(range(n), key=lambda i: pop[i].fitness, reverse=True)
    recv_raw, send_raw = [0.0] * n, [0.0] * n
    for pos, i in enumerate(order):
        recv_raw[i] = n - pos
        send_raw[i] = pos + 1
    return [x / sum(send_raw) for x in send_raw], [x / sum(recv_raw) for x in recv_raw]

def cultural_exchange(regions: List[Region], reqs: List[Request], fit: Fitness, prob: float = 0.25):
    send_r, recv_r = rank_exchange_rates(regions)
    per_region_rates = [(individual_ranks(reg.pop)) for reg in regions]
    for ri, reg in enumerate(regions):
        send_i, _ = per_region_rates[ri]
        for i, me in enumerate(reg.pop):
            if random.random() >= prob * send_r[ri] * send_i[i]: continue
            candidates, weights = [], []
            for rj, other in enumerate(regions):
                if rj == ri: continue
                recv_j = per_region_rates[rj][1]
                for k in range(len(other.pop)):
                    candidates.append((rj, k));
                    weights.append(recv_r[rj] * recv_j[k])
            if not candidates: continue
            rj, kj = candidates[random.choices(range(len(candidates)), weights=weights, k=1)[0]]
            child = crossover_constructive(reqs, me, regions[rj].pop[kj])
            fit.eval(child)
            if child.fitness >= me.fitness: reg.pop[i] = child

def effective_generations(pop_size: int, regions: int, fe_target: Optional[int]) -> Optional[int]:
    if not fe_target or fe_target <= 0: return None
    total_pop = pop_size * max(1, regions)
    return max(1, (fe_target - total_pop) // total_pop)

def annealed_probs(gargs: 'GAParams', gen: int, G: int) -> Tuple[float, float]:
    burn_in = int(gargs.burn_in_frac * G)
    if gen <= burn_in: return 0.0, 0.0
    t = (gen - burn_in) / max(1, G - burn_in)
    p_act = max(gargs.p_active_min, gargs.p_active0 * pow(1 - t, gargs.anneal_alpha))
    p_pas = max(gargs.p_passive_min, gargs.p_passive0 * pow(1 - t, gargs.anneal_beta))
    return p_act, p_pas

def diversity_hamming(pop: List[Individual]) -> float:
    N = len(pop)
    if N < 2 or not pop[0].genes: return 0.0
    s, cnt, L = 0.0, 0, len(pop[0].genes)
    for i in range(N):
        gi = pop[i].genes
        for j in range(i + 1, N):
            s += sum(1 for a, b in zip(gi, pop[j].genes) if a != b) / L
            cnt += 1
    return (2 * s) / (N * (N - 1)) if cnt else 0.0

def immigrate_if_needed(pop: List[Individual], fit: Fitness, reqs: List[Request], D_min: float, rho: float,
                        cooldown: int, elite_keep: int, gen: int, last_imm_gen: int) -> int:
    if (gen - last_imm_gen) <= cooldown or diversity_hamming(pop) >= D_min: return last_imm_gen
    m = max(2, round(rho * len(pop)))
    immigrants = []
    for _ in range(m):
        ind = random_feasible_individual(reqs)
        fit.eval(ind)
        immigrants.append(ind)
    pop.sort(key=lambda z: z.fitness, reverse=True)
    keep = max(0, min(elite_keep, len(pop) - m))
    pop[keep:keep + (len(pop) - keep - m)] = pop[keep:len(pop) - m]
    pop[-m:] = immigrants
    return gen

@dataclass
class GAParams:
    pop_size: int = 100
    regions: int = 10
    gens: int = 500
    pc: float = 0.85
    pm: float = 0.2
    elite_k: int = 10
    influence_period: int = 10
    exchange_period: int = 5
    parallel: bool = True
    workers: Optional[int] = None
    burn_in_frac: float = 0.10
    p_active0: float = 0.60
    p_passive0: float = 0.20
    p_active_min: float = 0.10
    p_passive_min: float = 0.05
    anneal_alpha: float = 2.0
    anneal_beta: float = 1.0
    fe_target: Optional[int] = None
    diversity_min: float = 0.15
    immigrants_rho: float = 0.05
    immigrants_cooldown: int = 5
    immigrants_elite_keep: int = 2
    use_fe_budget: bool = True
    use_learning: bool = True
    use_belief: bool = True
    use_exchange: bool = True
    use_immigrants: bool = True
    use_annealing: bool = True

def roulette(pop: List[Individual]) -> Individual:
    s = sum(max(0.0, ind.fitness) for ind in pop)
    if s <= 0: return random.choice(pop)
    r = random.random() * s
    acc = 0.0
    for ind in pop:
        acc += max(0.0, ind.fitness)
        if acc >= r: return ind
    return pop[-1]

def evolve_region_one_gen(ri: int, region_state: Region, reqs: List[Request], fit_params: FitnessParams,
                          gargs: GAParams, gen: int, seed: int) -> Tuple[int, Region, float]:
    random.seed(seed + 7919 * ri + 104729 * gen)
    fit = Fitness(reqs, fit_params)
    G_eff = effective_generations(gargs.pop_size, gargs.regions,
                                  gargs.fe_target) if gargs.use_fe_budget and gargs.fe_target else gargs.gens
    p_act, p_pas = annealed_probs(gargs, gen, G_eff) if gargs.use_annealing else (gargs.p_active0, gargs.p_passive0)

    reg = region_state
    new_pop: List[Individual] = []
    elites = sorted(reg.pop, key=lambda z: z.fitness, reverse=True)[: max(1, gargs.pop_size // 10)]
    new_pop.extend(x.clone() for x in elites)

    while len(new_pop) < gargs.pop_size:
        p1 = roulette(reg.pop)
        child = crossover_constructive(reqs, p1, roulette(reg.pop)) if random.random() < gargs.pc else p1.clone()
        child = mutate(reqs, child, gargs.pm)
        if gargs.use_learning:
            if random.random() < p_act:
                child = active_learning(reqs, child, elites[0], fit)
            elif random.random() < p_pas:
                child = passive_learning(reqs, child, roulette(reg.pop))
        fit.eval(child)
        new_pop.append(child)

    reg.pop = new_pop
    reg.belief.accept(reg.pop)
    return ri, reg, max(ind.fitness for ind in reg.pop)

def evolve_ce_slo_parallel(reqs: List[Request], fit: Fitness, gargs: GAParams, seed: int = 0, verbose: bool = False,
                           progress: bool = False) -> Tuple[Individual, List[float]]:
    random.seed(seed)
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
    G = effective_generations(gargs.pop_size, gargs.regions,
                              gargs.fe_target) if gargs.use_fe_budget and gargs.fe_target else gargs.gens
    workers = gargs.workers or min(gargs.regions, os.cpu_count() or 1)
    fit_params = fit.p
    last_imm_gen = [-10 ** 9] * len(regions)

    history = [best.fitness]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for gen in range(1, G + 1):
            futures = [ex.submit(evolve_region_one_gen, ri, regions[ri], reqs, fit_params, gargs, gen, seed) for ri in
                       range(len(regions))]
            results = [f.result() for f in futures]
            results.sort(key=lambda x: x[0])
            for ri, reg_state, _ in results: regions[ri] = reg_state

            if gargs.use_belief and gargs.influence_period and gen % gargs.influence_period == 0:
                for reg in regions:
                    reg.belief.influence(reg.pop)
                    for ind in reg.pop: fit.eval(ind)

            if gargs.use_exchange and gargs.exchange_period and gen % gargs.exchange_period == 0:
                cultural_exchange(regions, reqs, fit, prob=0.25)

            if gargs.use_immigrants:
                for ri, reg in enumerate(regions):
                    last_imm_gen[ri] = immigrate_if_needed(reg.pop, fit, reqs, gargs.diversity_min,
                                                           gargs.immigrants_rho, gargs.immigrants_cooldown,
                                                           gargs.immigrants_elite_keep, gen, last_imm_gen[ri])

            cur_best = max((ind for reg in regions for ind in reg.pop), key=lambda z: z.fitness)
            if cur_best.fitness > best.fitness: best = cur_best

            history.append(best.fitness)

            if progress: _print_progress("Evolving", gen, G)

    return best, history

def build_ce_params_from_ablation(base: GAParams, ablation: str, regions_default: int) -> GAParams:
    p = dataclasses.replace(base)
    p.use_learning = p.use_belief = p.use_exchange = p.use_immigrants = p.use_annealing = True
    p.regions = regions_default
    if ablation == "no_learning":
        p.use_learning = False
    elif ablation == "no_belief":
        p.use_belief = False
    elif ablation == "no_exchange":
        p.use_exchange = False
    elif ablation == "no_immigrants":
        p.use_immigrants = False
    elif ablation == "no_anneal":
        p.use_annealing = False
    elif ablation == "only_exchange":
        p.use_learning = p.use_belief = p.use_immigrants = False
    elif ablation == "only_belief":
        p.use_learning = p.use_exchange = p.use_immigrants = False
    elif ablation == "only_learning":
        p.use_belief = p.use_exchange = p.use_immigrants = False
    elif ablation == "regions1":
        p.regions = 1
    return p

def calc_statistics(data: List[float]) -> Tuple[float, float, float]:
    n = len(data)
    mean = sum(data) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1)) if n > 1 else 0.0
    ci = 2.262 * std / math.sqrt(n)  # t-value for 95% CI and df=9
    return mean, std, ci

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1997)
    parser.add_argument("--csv", type=str, default="../Init_Data/DataSet_weeks1_123.csv")
    parser.add_argument("--regions", type=int, default=10)
    parser.add_argument("--gens", type=int, default=500)
    parser.add_argument("--run-all-ablations", action="store_true", default=True,
                        help="Run 10 independent experiments for all variants")
    args = parser.parse_args()

    ALL_ABLATIONS = ["full", "no_learning", "no_belief", "no_exchange", "no_immigrants", "only_exchange",
                     "only_belief", "only_learning", "regions1"]
    NUM_RUNS = 10

    try:
        reqs = load_requests_from_csv(args.csv)
        print(f"Loaded {len(reqs)} requests from {args.csv}")
    except Exception:
        reqs = make_demo_requests(seed=args.seed)
        print("Using demo requests.")

    fit = Fitness(reqs, FitnessParams())
    base_params = GAParams(parallel=True, use_fe_budget=False, gens=args.gens)

    final_stats = {}
    mean_curves = {ab: [0.0] * (args.gens + 1) for ab in ALL_ABLATIONS}

    for ab in ALL_ABLATIONS:
        print(f"\n=== Running {ab} ({NUM_RUNS} runs) ===")
        params = build_ce_params_from_ablation(base_params, ab, args.regions)

        run_fitnesses = []
        run_histories = []

        for run in range(NUM_RUNS):
            current_seed = args.seed + run
            print(f"Run {run + 1}/{NUM_RUNS} for {ab}...")
            best, history = evolve_ce_slo_parallel(reqs, fit, params, seed=current_seed, progress=False)

            run_fitnesses.append(best.fitness)
            run_histories.append(history)

        mean_fit, std_fit, ci_fit = calc_statistics(run_fitnesses)
        final_stats[ab] = (mean_fit, std_fit, ci_fit)

        for gen in range(args.gens + 1):
            mean_curves[ab][gen] = sum(rh[gen] for rh in run_histories) / NUM_RUNS

    print("\n" + "=" * 50)
    print(f"Statistical Summary over {NUM_RUNS} Independent Runs")
    print("=" * 50)
    print(f"{'Algorithm':<15} | {'Mean':<8} | {'Std Dev':<8} | {'95% CI':<8}")
    print("-" * 50)

    with open("ablation_statistics_for_exited_excel.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Mean Fitness", "Standard Deviation", "95% Confidence Interval"])
        for ab in ALL_ABLATIONS:
            mean_fit, std_fit, ci_fit = final_stats[ab]
            print(f"{ab:<15} | {mean_fit:.5f} | {std_fit:.5f} | ±{ci_fit:.5f}")
            writer.writerow([ab, mean_fit, std_fit, ci_fit])

    with open("ablation_mean_curves_for_exited_excel.csv", "w", newline='') as f:
        writer = csv.writer(f)
        header = ["Iteration"] + ALL_ABLATIONS
        writer.writerow(header)
        for gen in range(args.gens + 1):
            row = [gen] + [mean_curves[ab][gen] for ab in ALL_ABLATIONS]
            writer.writerow(row)

    print("\n[INFO] 'ablation_statistics.csv' and 'ablation_mean_curves.csv' have been generated.")