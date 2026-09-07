from __future__ import annotations

"""
SOVEREIGN_MNQ_V5S_SHORT_ROUTER_RESEARCH
Research-only. Sends no orders.

Frozen rule selected before 2025 was opened:
- TREND_FVG_CONTINUATION only
- SHORT only
- 10:00 <= signal time ET < 11:00
- V3 grade cap >= 4
- 1.0 <= relative volume < 2.0

Protocol: 2020-2023 development, 2024 selection, 2025 locked holdout.
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
    if c.family != "TREND_FVG_CONTINUATION" or c.direction >= 0:
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
    return cap >= 4 and 1.0 <= vr < 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--source-tz", default="UTC")
    ap.add_argument("--input-minutes", type=int, choices=[1,5], default=5)
    ap.add_argument("--out", type=Path, default=Path("MNQ_V5S_RESULTS"))
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
    report["engine"] = "SOVEREIGN_MNQ_V5S_SHORT_ROUTER_RESEARCH"
    report["router"] = {
        "family":"TREND_FVG_CONTINUATION",
        "direction":"SHORT",
        "entry_window_et":"10:00-11:00",
        "min_grade_cap":4,
        "vol_ratio_range":"[1.0,2.0)",
        "selection_protocol":"2020-2023 develop; 2024 select; 2025 locked holdout",
    }
    trades.to_csv(args.out/"trades.csv", index=False)
    pd.DataFrame([asdict(c) for c in cands]).to_csv(args.out/"candidates.csv", index=False)
    (args.out/"report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
