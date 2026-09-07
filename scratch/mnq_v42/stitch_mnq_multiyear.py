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
        meta.append({
            "date": dt,
            "contract": p.parent.name,
            "path": str(p),
            "rows": int(len(df)),
            "volume": float(df["volume"].sum()),
            "first_ts": str(df["datetime"].min()),
            "last_ts": str(df["datetime"].max()),
        })
        if i % 500 == 0:
            print("scanned", i, "of", len(paths), flush=True)

    m = pd.DataFrame(meta)
    if m.empty:
        raise SystemExit("No usable files in requested date range")
    m["contract_key"] = m["contract"].map(contract_sort_key)
    m = m.sort_values(["contract", "date"])

    # Causal roll choice: today's contract is ranked by that contract's most recent
    # completed prior-day Last-trade volume. Today's eventual volume is never used.
    m["prev_volume"] = m.groupby("contract", sort=False)["volume"].shift(1)
    m["prev_date"] = m.groupby("contract", sort=False)["date"].shift(1)
    m["prev_age_days"] = (m["date"] - m["prev_date"]).dt.days
    # Stale prior observations should not decide a roll after long contract inactivity.
    m.loc[m["prev_age_days"].gt(7), "prev_volume"] = pd.NA

    chosen_rows = []
    for dt, g in m.groupby("date", sort=True):
        ranked = g[g["prev_volume"].notna()].copy()
        if not ranked.empty:
            ranked = ranked.sort_values(["prev_volume", "contract_key"], ascending=[False, False])
            row = ranked.iloc[0].copy()
            row["selection_basis"] = "prior_completed_contract_day_volume"
        else:
            # Initial warmup only: no prior completed volume exists. Prefer the nearest
            # listed expiry (earliest contract_key) rather than looking at today's volume.
            ranked = g.sort_values(["contract_key"], ascending=[True])
            row = ranked.iloc[0].copy()
            row["selection_basis"] = "no_prior_volume_nearest_expiry_fallback"
        chosen_rows.append(row)

    chosen = pd.DataFrame(chosen_rows).sort_values("date").copy()
    chosen["date"] = chosen["date"].dt.date.astype(str)
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
    dupes = int(x["datetime"].duplicated().sum())
    if dupes:
        print("WARNING duplicate timestamps", dupes)
        x = x.drop_duplicates("datetime", keep="last")

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
        "selection_rule": "today uses each contract's most recent completed prior-day total Last-trade volume; stale >7d ignored; no-prior fallback uses nearest expiry",
        "source_repo": "mbytes21/MNQ_DATA",
    }
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    main()
