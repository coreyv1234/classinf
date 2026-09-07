from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import pandas as pd


def contract_sort_key(name: str):
    m = re.search(r"MNQ\s+(\d{2})-(\d{2})", name)
    if not m:
        return (9999, 99)
    month, yy = map(int, m.groups())
    return (2000 + yy, month)


def read_one(path: Path):
    try:
        df = pd.read_csv(path, usecols=["datetime", "open", "high", "low", "close", "volume"])
    except Exception:
        return None
    if df.empty:
        return None
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    if df.empty:
        return None
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--start", default="2019-12-01")
    ap.add_argument("--end", default="2025-12-31 23:59:59")
    ap.add_argument("--out", default="MNQ_STITCHED")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    paths = sorted(root.glob("plaintext_csv/MNQ */*.Last.csv"))
    if not paths:
        raise SystemExit(f"No Last.csv files found under {root}")

    meta = []
    for i, p in enumerate(paths, 1):
        date_match = re.match(r"(\d{8})\.Last\.csv$", p.name)
        if not date_match:
            continue
        dt = pd.to_datetime(date_match.group(1), format="%Y%m%d", errors="coerce")
        if pd.isna(dt) or dt < start.normalize() or dt > end.normalize():
            continue
        df = read_one(p)
        if df is None:
            continue
        total_vol = float(df["volume"].sum())
        meta.append({
            "date": dt.date().isoformat(),
            "contract": p.parent.name,
            "path": str(p),
            "rows": int(len(df)),
            "volume": total_vol,
            "first_ts": str(df["datetime"].min()),
            "last_ts": str(df["datetime"].max()),
        })
        if i % 500 == 0:
            print("scanned", i, "of", len(paths), flush=True)

    m = pd.DataFrame(meta)
    if m.empty:
        raise SystemExit("No usable files in requested date range")
    m["contract_key"] = m["contract"].map(contract_sort_key)
    # Select exactly one contract per calendar date, using total Last-trade volume.
    # Tie-break toward the later contract to avoid hanging onto an expiring contract.
    m = m.sort_values(["date", "volume", "contract_key"], ascending=[True, False, False])
    chosen = m.drop_duplicates("date", keep="first").copy()
    chosen.to_csv(out / "selected_contract_by_day.csv", index=False)

    frames = []
    for j, row in enumerate(chosen.itertuples(index=False), 1):
        df = read_one(Path(row.path))
        if df is None:
            continue
        df["contract"] = row.contract
        frames.append(df)
        if j % 250 == 0:
            print("loaded selected days", j, "of", len(chosen), flush=True)

    x = pd.concat(frames, ignore_index=True)
    x = x[(x["datetime"] >= start) & (x["datetime"] <= end)].copy()
    x = x.sort_values("datetime")
    # One selected contract per date means duplicate timestamps should be rare; fail visibly if they exist.
    dupes = int(x["datetime"].duplicated().sum())
    if dupes:
        print("WARNING duplicate timestamps", dupes)
        x = x.drop_duplicates("datetime", keep="last")

    # Source files appear in UTC-like exchange timestamps; preserve naive timestamps and tell the runner UTC.
    x = x.set_index("datetime")
    bars = x.resample("5min", origin="start_day", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open", "high", "low", "close"]).reset_index()

    bars.to_csv(out / "MNQ_5m_2019_2025.csv", index=False)
    coverage = {
        "source_files_seen": int(len(paths)),
        "usable_contract_day_files": int(len(m)),
        "selected_calendar_days": int(len(chosen)),
        "minute_rows": int(len(x)),
        "bars_5m": int(len(bars)),
        "first_minute": str(x.index.min()),
        "last_minute": str(x.index.max()),
        "first_5m": str(bars["datetime"].min()),
        "last_5m": str(bars["datetime"].max()),
        "duplicate_minutes_removed": dupes,
        "selection_rule": "highest total Last-trade volume per calendar date; later contract wins ties",
        "source_repo": "mbytes21/MNQ_DATA",
    }
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    main()
