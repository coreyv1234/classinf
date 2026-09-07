from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def efficiency(close: pd.Series, n: int) -> pd.Series:
    move = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return move / path.replace(0, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bars_csv')
    ap.add_argument('trades_csv')
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    b = pd.read_csv(args.bars_csv)
    b['dt_utc'] = pd.to_datetime(b['datetime'], utc=True, errors='coerce')
    for c in ['open','high','low','close','volume']:
        b[c] = pd.to_numeric(b[c], errors='coerce')
    b = b.dropna(subset=['dt_utc','open','high','low','close']).sort_values('dt_utc').reset_index(drop=True)
    et = b['dt_utc'].dt.tz_convert('America/New_York')
    b['et_date'] = et.dt.date
    b['et_hour'] = et.dt.hour
    b['et_minute'] = et.dt.minute
    # CME equity-index session is conventionally 18:00 ET through 17:00 ET next day.
    sess = et.dt.normalize()
    sess = sess.where(et.dt.hour < 18, sess + pd.Timedelta(days=1))
    b['cme_session'] = sess.dt.date

    prev_close = b['close'].shift(1)
    tr = pd.concat([
        b['high'] - b['low'],
        (b['high'] - prev_close).abs(),
        (b['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean().shift(1)
    atr50 = tr.rolling(50, min_periods=50).mean().shift(1)
    atr_med = atr14.rolling(1000, min_periods=250).median().shift(1)
    b['r_atr14'] = atr14
    b['r_atr50'] = atr50
    b['r_atr_ratio'] = atr14 / atr_med.replace(0, np.nan)

    for span in [20,50,200]:
        ema = b['close'].ewm(span=span, adjust=False).mean().shift(1)
        b[f'r_ema{span}_dist'] = (prev_close - ema) / atr14.replace(0, np.nan)
        lag = {20:12,50:24,200:48}[span]
        b[f'r_ema{span}_slope'] = (ema - ema.shift(lag)) / (lag * atr14.replace(0, np.nan))

    for n in [3,12,48,96,288]:
        b[f'r_ret_{n}'] = prev_close.pct_change(n)
    for n in [12,48,96]:
        b[f'r_eff_{n}'] = efficiency(b['close'].shift(1), n)

    range_now = b['high'] - b['low']
    b['r_bar_range_atr'] = range_now.shift(1) / atr14.replace(0, np.nan)
    vol_med = b['volume'].rolling(50, min_periods=20).median().shift(1)
    b['r_volume_ratio'] = b['volume'].shift(1) / vol_med.replace(0, np.nan)

    # Running current-session state, all causal to the bar close.
    grp = b.groupby('cme_session', sort=False)
    run_hi = grp['high'].cummax()
    run_lo = grp['low'].cummin()
    sess_open = grp['open'].transform('first')
    den = (run_hi - run_lo).replace(0, np.nan)
    b['r_session_loc'] = ((prev_close - run_lo.shift(1)) / (run_hi.shift(1) - run_lo.shift(1)).replace(0, np.nan)).clip(-1,2)
    b['r_from_session_open_atr'] = (prev_close - sess_open) / atr14.replace(0, np.nan)

    # Completed-session features. Shift one full session before mapping to current bars.
    daily = grp.agg(
        d_open=('open','first'), d_high=('high','max'), d_low=('low','min'),
        d_close=('close','last'), d_volume=('volume','sum')
    ).reset_index()
    daily['d_range'] = daily['d_high'] - daily['d_low']
    daily['d_ret1'] = daily['d_close'].pct_change(1)
    daily['d_ret5'] = daily['d_close'].pct_change(5)
    daily['d_ret20'] = daily['d_close'].pct_change(20)
    daily['d_range_rel20'] = daily['d_range'] / daily['d_range'].rolling(20, min_periods=10).median().replace(0, np.nan)
    daily['d_vol_rel20'] = daily['d_volume'] / daily['d_volume'].rolling(20, min_periods=10).median().replace(0, np.nan)
    daily['d_close_loc'] = (daily['d_close']-daily['d_low'])/(daily['d_high']-daily['d_low']).replace(0,np.nan)
    dcols=['d_close','d_high','d_low','d_ret1','d_ret5','d_ret20','d_range_rel20','d_vol_rel20','d_close_loc']
    for c in dcols:
        daily['r_prev_'+c] = daily[c].shift(1)
    joincols=['cme_session']+['r_prev_'+c for c in dcols]
    b=b.merge(daily[joincols],on='cme_session',how='left')
    b['r_dist_prev_high_atr']=(prev_close-b['r_prev_d_high'])/atr14.replace(0,np.nan)
    b['r_dist_prev_low_atr']=(prev_close-b['r_prev_d_low'])/atr14.replace(0,np.nan)
    b['r_dist_prev_close_atr']=(prev_close-b['r_prev_d_close'])/atr14.replace(0,np.nan)

    clock = b['et_hour'] + b['et_minute']/60.0
    b['r_tod_sin'] = np.sin(2*np.pi*clock/24.0)
    b['r_tod_cos'] = np.cos(2*np.pi*clock/24.0)

    regime_cols=[c for c in b.columns if c.startswith('r_')]
    right=b[['dt_utc']+regime_cols].sort_values('dt_utc')

    t=pd.read_csv(args.trades_csv)
    t['signal_utc']=pd.to_datetime(t['signal_time'],utc=True,errors='coerce')
    t['session_date_parsed']=pd.to_datetime(t['session_date'],errors='coerce')
    t=t[t['session_date_parsed'].dt.year.eq(args.year)].copy().sort_values('signal_utc')
    ann=pd.merge_asof(t,right,on='dt_utc' if False else 'signal_utc',right_on='dt_utc',direction='backward',tolerance=pd.Timedelta('10min'))
    # merge_asof with different key names retains right key; report missing features visibly.
    miss=float(ann[regime_cols].isna().all(axis=1).mean()) if len(ann) else 0.0
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    ann.to_csv(out,index=False)
    print({'year':args.year,'trades':len(ann),'regime_cols':len(regime_cols),'all_regime_missing_rate':miss,'first_signal':str(ann['signal_utc'].min()),'last_signal':str(ann['signal_utc'].max())})


if __name__=='__main__':
    main()
