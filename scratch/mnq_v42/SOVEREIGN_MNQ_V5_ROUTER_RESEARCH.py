from __future__ import annotations

"""
SOVEREIGN_MNQ_V5_ROUTER_RESEARCH
Research-only MNQ router. Sends no orders.

Rule was selected without 2025/2026:
- Development: 2020-2023
- Selection: 2024
- Holdout: 2025

Only allows V3 TREND_FVG_CONTINUATION candidates when:
- signal time is 10:00 <= ET < 11:00
- V3 setup grade is 4-cap (high-quality trend FVG)
- relative volume is below 2.0

This is intentionally simple to reduce selection degrees of freedom.
"""
import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import SOVEREIGN_MNQ_V3_BIG_R_GRADED_RESEARCH as v3

base = v3.base
ET = v3.ET


def router_accept(x: pd.DataFrame, c: base.Candidate) -> bool:
    if c.family != "TREND_FVG_CONTINUATION":
        return False
    t = c.signal_time.tz_convert(ET).time()
    if not (pd.Timestamp("10:00").time() <= t < pd.Timestamp("11:00").time()):
        return False
    sig = x.loc[x.bar_end == c.signal_time]
    if sig.empty:
        return False
    r = sig.iloc[-1]
    vr = float(r.vol_ratio) if np.isfinite(r.vol_ratio) else np.nan
    ts = float(r.trend_strength) if np.isfinite(r.trend_strength) else np.nan
    if not np.isfinite(vr) or not np.isfinite(ts):
        return False
    cap = v3.setup_cap(c.family, vr, ts)
    return cap >= 4 and vr < 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--source-tz", default="UTC")
    ap.add_argument("--input-minutes", type=int, choices=[1,5], default=5)
    ap.add_argument("--out", type=Path, default=Path("MNQ_V5_RESULTS"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = base.Config()
    x = base.load_bars(args.csv, args.source_tz, args.input_minutes)
    x, _ = base.add_features(x, cfg)
    x = base.add_session_levels(x)
    x = v3.add_v3_features(x)
    zones5 = base.detect_zones(x, "5m", cfg)
    all_cands = v3.generate_v3_candidates(x, zones5, cfg)
    cands = [c for c in all_cands if router_accept(x, c)]
    trades = v3.run_v3(x, cands, cfg)
    report = v3.build_report(x, cands, trades, cfg)
    report["engine"] = "SOVEREIGN_MNQ_V5_ROUTER_RESEARCH"
    report["router"] = {
        "family":"TREND_FVG_CONTINUATION",
        "entry_window_et":"10:00-11:00",
        "min_grade_cap":4,
        "vol_ratio_max_exclusive":2.0,
        "selection_protocol":"2020-2023 develop; 2024 select; 2025 holdout",
    }
    trades.to_csv(args.out/"trades.csv", index=False)
    pd.DataFrame([asdict(c) for c in cands]).to_csv(args.out/"candidates.csv", index=False)
    (args.out/"report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
