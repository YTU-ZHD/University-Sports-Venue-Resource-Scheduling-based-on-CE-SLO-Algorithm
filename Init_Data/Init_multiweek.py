import argparse
import numpy as np
import pandas as pd

def build_args():
    ap = argparse.ArgumentParser(
        description="Generate synthetic multi-week requests with nested reproducibility."
    )
    ap.add_argument("--weeks", type=int, default=1, help="Number of weeks (>=1).")
    ap.add_argument("--seed", type=int, default=123, help="Base random seed.")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path.")
    ap.add_argument("--format", type=str, default=None, choices=["legacy", "research"],
                    help="legacy: RequestID,Type,DesiredDay,DesiredSlot,Duration; research: id,type,preferred_day,preferred_start,duration")
    ap.add_argument("--with-court", action="store_true",
                    help="Include preferred_court column in research format (default: off).")
    return ap.parse_args()

WEEKLY_COUNTS = {
    "Hard Game": 8,
    "Soft Game": 12,
    "Teaching": 60,
    "Student Club": 40,
    "Free Exercise": 50,
}
SLOTS_PER_DAY = 12
DAY_IDX_TO_NAME = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

WEEKDAY_ALLOWED = {
    "Hard Game":      list(range(7)),
    "Soft Game":      list(range(7)),
    "Teaching":       [0, 1, 2, 3, 4],
    "Student Club":   list(range(7)),
    "Free Exercise":  [5, 6],
}

def sample_duration(rtype: str, rng: np.random.Generator) -> int:
    if rtype == "Hard Game":
        return int(rng.integers(3, 5))   # 3-4
    if rtype == "Soft Game":
        return int(rng.integers(3, 5))   # 3-4
    if rtype == "Teaching":
        return 2
    if rtype == "Student Club":
        return int(rng.integers(1, 4))   # 1-3
    if rtype == "Free Exercise":
        return 1
    raise ValueError("Unknown type")

def generate_records(weeks: int, seed: int):
    ss = np.random.SeedSequence(seed)
    child_ss = ss.spawn(weeks)
    week_rngs = [np.random.default_rng(s) for s in child_ss]

    records = []
    rid = 1
    for w in range(weeks):
        rng = week_rngs[w]
        base_day = w * 7
        for rtype, weekly_cnt in WEEKLY_COUNTS.items():
            for _ in range(weekly_cnt):
                wd = int(rng.choice(WEEKDAY_ALLOWED[rtype]))
                day = base_day + wd
                dur = sample_duration(rtype, rng)
                start = int(rng.integers(0, SLOTS_PER_DAY - dur + 1))   # 0..11
                records.append((rid, rtype, day, start, dur))
                rid += 1
    return records

def to_legacy_df(records):
    rows = []
    for rid, rtype, day, start, dur in records:
        rows.append({
            "RequestID": rid,
            "Type": rtype,
            "DesiredDay": DAY_IDX_TO_NAME[day % 7],
            "DesiredSlot": start + 1,
            "Duration": dur,
        })
    return pd.DataFrame(rows)

def to_research_df(records, with_court: bool):
    MAP = {
        "Hard Game": "HARD",
        "Soft Game": "SOFT",
        "Teaching": "TEACHING",
        "Student Club": "CLUB",
        "Free Exercise": "FREE",
    }
    rows = []
    for rid, rtype, day, start, dur in records:
        row = {
            "id": rid,
            "type": MAP[rtype],
            "preferred_day": day,
            "preferred_start": start,
            "duration": dur,
        }
        if with_court:
            row["preferred_court"] = ""
        rows.append(row)
    df = pd.DataFrame(rows)

    if with_court:
        df = df[["id","type","preferred_day","preferred_start","preferred_court","duration"]]
    else:
        df = df[["id","type","preferred_day","preferred_start","duration"]]
    return df

def main():
    args = build_args()
    if args.weeks < 1:
        raise ValueError("weeks must be >= 1")
    records = generate_records(args.weeks, args.seed)

    fmt = args.format if args.format is not None else ("legacy" if args.weeks == 1 else "research")

    if fmt == "legacy" and args.weeks != 1:
        raise ValueError("The legacy format only supports weeks=1; for multiple weeks, please use the research format.")

    if fmt == "legacy":
        df = to_legacy_df(records)
        out = args.out or f"DataSet_{args.seed}.csv"
    else:
        df = to_research_df(records, with_court=args.with_court)
        out = args.out or f"DataSet_weeks{args.weeks}.csv"

    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {out} (format={fmt}, weeks={args.weeks}, seed={args.seed})")

if __name__ == "__main__":
    main()
