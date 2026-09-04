from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CT='America/Chicago'
SRC=Path('mgc-source/aggregated/continuous/ohlcv_MGC_5m_continuous.csv')
OUT=Path('mgc_research/v12_results'); OUT.mkdir(parents=True, exist_ok=True)
COST_R=0.06
HORIZON=8
SESSION_START=120
SESSION_LAST=915
MAX_TRADES_DAY=5
DAY_STOP_R=2.15
DIR_LOSS_KILL=2
RISK_RESEARCH=225.0
RISK_SAFE=150.0
STOP_MULTS=[0.75,0.90,1.05]
RRS=[2.0,2.5,3.0]
THRESHOLDS=np.round(np.arange(0.48,0.741,0.02),2)

FEATURES=['gap20_50','ema9_d','vwap_d','loc12','loc20','loc60','body_atr','range_atr','close_loc','ret1_atr','ret3_atr','ret6_atr','volz','rv_ratio','er10','sweep_hi','sweep_lo','vwap_cross_up','vwap_cross_dn','sin_t','cos_t','dow_sin','dow_cos']

def load():
    x=pd.read_csv(SRC)
    tcol='timestamp' if 'timestamp' in x.columns else ('datetime' if 'datetime' in x.columns else x.columns[0])
    x['ts']=pd.to_datetime(x[tcol],utc=True,errors='coerce')
    x=x.dropna(subset=['ts']).set_index('ts').sort_index(); x=x[~x.index.duplicated(keep='last')]
    for c in ['open','high','low','close','volume']: x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna(subset=['open','high','low','close','volume'])

def feats(x):
    x=x.copy(); ct=x.index.tz_convert(CT)
    x['date']=ct.date; x['year']=ct.year; x['minute']=ct.hour*60+ct.minute; x['dow']=ct.dayofweek
    o,h,l,c,v=[x[k].astype(float) for k in ['open','high','low','close','volume']]
    pc=c.shift(); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    x['atr']=tr.ewm(alpha=1/14,adjust=False).mean().clip(lower=.05)
    e9=c.ewm(span=9,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    x['ema9_d']=(c-e9)/(x.atr+1e-9); x['gap20_50']=(e20-e50)/(x.atr+1e-9)
    for n in [12,20,60]:
        rh=h.rolling(n).max(); rl=l.rolling(n).min(); x[f'loc{n}']=(c-rl)/(rh-rl+1e-9)
    rhp=h.shift(1).rolling(12).max(); rlp=l.shift(1).rolling(12).min()
    x['sweep_hi']=((h>rhp+.03*x.atr)&(c<rhp)).astype(float)
    x['sweep_lo']=((l<rlp-.03*x.atr)&(c>rlp)).astype(float)
    x['body_atr']=(c-o)/(x.atr+1e-9); x['range_atr']=(h-l)/(x.atr+1e-9); x['close_loc']=(c-l)/(h-l+1e-9)
    x['ret1_atr']=(c-c.shift(1))/(x.atr+1e-9); x['ret3_atr']=(c-c.shift(3))/(x.atr+1e-9); x['ret6_atr']=(c-c.shift(6))/(x.atr+1e-9)
    vm=v.rolling(20).mean(); vs=v.rolling(20).std(); x['volz']=(v-vm)/(vs+1e-9)
    r=c.pct_change(); x['rv_ratio']=r.rolling(6).std()/(r.rolling(24).std()+1e-12)
    x['er10']=(c-c.shift(10)).abs()/(c.diff().abs().rolling(10).sum()+1e-9)
    typ=(h+l+c)/3; x['vwap']=(typ*v).groupby(x.date).cumsum()/v.groupby(x.date).cumsum().replace(0,np.nan)
    x['vwap_d']=(c-x.vwap)/(x.atr+1e-9)
    x['vwap_cross_up']=((c.shift(1)<=x.vwap.shift(1))&(c>x.vwap)).astype(float)
    x['vwap_cross_dn']=((c.shift(1)>=x.vwap.shift(1))&(c<x.vwap)).astype(float)
    x['sin_t']=np.sin(2*np.pi*x.minute/1440.0); x['cos_t']=np.cos(2*np.pi*x.minute/1440.0)
    x['dow_sin']=np.sin(2*np.pi*x.dow/5.0); x['dow_cos']=np.cos(2*np.pi*x.dow/5.0)
    return x

def candidates(x):
    good=np.isfinite(x[FEATURES+['atr']]).all(axis=1)
    base=(x.minute>=SESSION_START)&(x.minute<=SESSION_LAST)&good
    rows=[]
    # Broad but location-aware universe. A short cannot be initiated near a 20-bar floor; a long cannot be initiated near a 20-bar ceiling.
    # This is an explicit structural anti-chase gate, not a learned threshold.
    for side in ['LONG','SHORT']:
        if side=='LONG':
            m=base&(x.loc20<=.76)&((x.close_loc>=.38)|(x.sweep_lo>0)|(x.vwap_cross_up>0))
        else:
            m=base&(x.loc20>=.24)&((x.close_loc<=.62)|(x.sweep_hi>0)|(x.vwap_cross_dn>0))
        inds=np.flatnonzero(m.to_numpy())
        q=x.iloc[inds][FEATURES+['date','year','minute','atr']].copy(); q['i']=inds; q['side']=side
        rows.append(q.reset_index(names='signal_ts'))
    z=pd.concat(rows,ignore_index=True).sort_values(['i','side']).reset_index(drop=True)
    return z

def label(x,c,stop_mult,rr):
    z=c.copy(); idx=z.i.to_numpy(int); ep=idx+1; valid=ep<len(x); z=z.loc[valid].copy(); idx=z.i.to_numpy(int); ep=idx+1
    op=x.open.to_numpy(float); hi=x.high.to_numpy(float); lo=x.low.to_numpy(float); day=pd.factorize(x.date)[0]
    atr=z.atr.to_numpy(float); side=z.side.to_numpy(); en=op[ep]; dist=stop_mult*atr
    stop=np.where(side=='LONG',en-dist,en+dist); target=np.where(side=='LONG',en+rr*dist,en-rr*dist)
    offs=np.arange(HORIZON,dtype=int); wi=ep[:,None]+offs[None,:]; inrange=wi<len(x); wi=np.minimum(wi,len(x)-1)
    same=inrange&(day[wi]==day[ep][:,None]); wh=hi[wi]; wl=lo[wi]; islong=(side=='LONG')[:,None]
    stophit=np.where(islong,wl<=stop[:,None],wh>=stop[:,None])&same
    targhit=np.where(islong,wh>=target[:,None],wl<=target[:,None])&same
    sentinel=HORIZON+1; fs=np.where(stophit.any(axis=1),stophit.argmax(axis=1),sentinel); ft=np.where(targhit.any(axis=1),targhit.argmax(axis=1),sentinel)
    # same-bar target+stop is scored as a loss because intrabar sequence is unknowable on 5m OHLC.
    win=ft<fs; exoff=np.minimum(np.minimum(fs,ft),HORIZON-1)
    z['entry_i']=ep; z['exit_i']=ep+exoff; z['entry']=en; z['stop']=stop; z['target']=target; z['y']=win.astype(int); z['netR']=np.where(win,rr-COST_R,-1.0-COST_R)
    return z

def side_features(d,side):
    a=d[FEATURES].copy()
    # directional transforms let the learner reason about trend alignment and chase consistently by side.
    s=1.0 if side=='LONG' else -1.0
    a['dir_gap']=s*d.gap20_50; a['dir_vwap']=s*d.vwap_d; a['dir_ret1']=s*d.ret1_atr; a['dir_ret3']=s*d.ret3_atr; a['dir_ret6']=s*d.ret6_atr
    a['chase_loc']=d.loc20 if side=='LONG' else 1.0-d.loc20
    a['dir_close_loc']=d.close_loc if side=='LONG' else 1.0-d.close_loc
    a['dir_sweep']=d.sweep_lo if side=='LONG' else d.sweep_hi
    a['dir_vwap_cross']=d.vwap_cross_up if side=='LONG' else d.vwap_cross_dn
    return a

def fit_model(train,side):
    d=train[train.side==side]
    X=side_features(d,side); y=d.y.to_numpy(int)
    model=HistGradientBoostingClassifier(max_iter=170,learning_rate=.045,max_depth=4,max_leaf_nodes=24,min_samples_leaf=90,l2_regularization=8.0,random_state=71)
    model.fit(X,y); return model

def add_probs(model,d,side):
    q=d[d.side==side].copy(); q['prob']=model.predict_proba(side_features(q,side))[:,1]; return q

def pf(s):
    s=np.asarray(s,float); gp=s[s>0].sum(); gl=-s[s<=0].sum(); return float(gp/gl) if gl>0 else 99.0

def simulate(d,tl,ts):
    q=d[((d.side=='LONG')&(d.prob>=tl))|((d.side=='SHORT')&(d.prob>=ts))].copy()
    q=q.sort_values(['entry_i','prob'],ascending=[True,False])
    out=[]; active_until=-1; dayp=defaultdict(float); dayn=defaultdict(int); dl=defaultdict(lambda:defaultdict(int))
    last_signal_side=set()
    for _,r in q.iterrows():
        ei=int(r.entry_i); date=r.date; side=r.side
        if ei<=active_until: continue
        if dayn[date]>=MAX_TRADES_DAY or dayp[date]<=-DAY_STOP_R or dl[date][side]>=DIR_LOSS_KILL: continue
        key=(int(r.i),side)
        if key in last_signal_side: continue
        last_signal_side.add(key)
        nr=float(r.netR); out.append({'signal_ts':r.signal_ts,'date':date,'year':int(r.year),'side':side,'prob':float(r.prob),'entry_i':ei,'exit_i':int(r.exit_i),'entry':float(r.entry),'stop':float(r.stop),'target':float(r.target),'netR':nr})
        dayp[date]+=nr; dayn[date]+=1
        if nr<0: dl[date][side]+=1
        active_until=int(r.exit_i)
    return pd.DataFrame(out)

def metrics(t,year=None):
    if t is None or len(t)==0:return {'trades':0}
    s=t.netR.to_numpy(float); daily=t.groupby('date').netR.sum(); gp=s[s>0].sum(); gl=-s[s<=0].sum()
    res={'trades':int(len(t)),'active_days':int(len(daily)),'tpd_active':float(len(t)/len(daily)),'netR':float(s.sum()),'PF':float(gp/gl) if gl>0 else 99.0,'win_pct':float((s>0).mean()*100),'green_pct':float((daily>0).mean()*100),'avg_active_day_R':float(daily.mean()),'worst_day_R':float(daily.min()),'best_day_R':float(daily.max())}
    for side in ['LONG','SHORT']:
        z=t[t.side==side]; res[side.lower()]={'trades':int(len(z)),'netR':float(z.netR.sum()) if len(z) else 0.0,'PF':pf(z.netR) if len(z) else 0.0}
    res['research_dollars']={'avg_active_day':res['avg_active_day_R']*RISK_RESEARCH,'worst_day':res['worst_day_R']*RISK_RESEARCH,'net':res['netR']*RISK_RESEARCH}
    res['safe_dollars']={'avg_active_day':res['avg_active_day_R']*RISK_SAFE,'worst_day':res['worst_day_R']*RISK_SAFE,'net':res['netR']*RISK_SAFE}
    return res

def auc_score(d):
    if len(d)==0 or d.y.nunique()<2:return None
    return float(roc_auc_score(d.y,d.prob))

def score_val(m):
    if m.get('trades',0)<100:return -1e9
    if m['long']['trades']<25 or m['short']['trades']<25:return -1e9
    if m['long']['netR']<=0 or m['short']['netR']<=0:return -1e9
    if m['PF']<1.25:return -1e9
    # reward robust expectancy and PF, penalize too few trades and oversized bad days
    return m['avg_active_day_R']+.32*min(m['PF'],3.0)+.0035*m['green_pct']-.10*max(0,abs(m['worst_day_R'])-3.1)-.03*abs(m['tpd_active']-2.8)

def bootstrap(t,n=2500,seed=71):
    if len(t)==0:return {}
    d=t.groupby('date').netR.sum().to_numpy(); rng=np.random.default_rng(seed); av=[]; dd=[]
    for _ in range(n):
        s=rng.choice(d,size=len(d),replace=True); av.append(float(np.mean(s))); eq=np.cumsum(s); peak=np.maximum.accumulate(np.r_[0.0,eq])[1:]; dd.append(float(np.max(peak-eq)))
    return {'avg_day_R_p05':float(np.quantile(av,.05)),'avg_day_R_median':float(np.median(av)),'avg_day_R_p95':float(np.quantile(av,.95)),'maxDD_R_median':float(np.median(dd)),'maxDD_R_p95':float(np.quantile(dd,.95))}

def main():
    x=feats(load()); base=candidates(x)
    all_grid=[]; bundles={}
    for sm in STOP_MULTS:
      for rr in RRS:
        z=label(x,base,sm,rr)
        # independent 2024 shadow test: train on 2023 only. It is diagnostic, not threshold tuning.
        y23=z[z.year==2023]
        shadow_parts=[]
        for side in ['LONG','SHORT']:
            m=fit_model(y23,side); shadow_parts.append(add_probs(m,z[z.year==2024],side))
        shadow=pd.concat(shadow_parts,ignore_index=True)
        shadow_auc={'long':auc_score(shadow[shadow.side=='LONG']),'short':auc_score(shadow[shadow.side=='SHORT'])}
        # final learner uses 2023-2024 only. Thresholds are chosen only on 2025.
        train=z[z.year<=2024]
        val=z[z.year==2025]
        hold=z[z.year==2026]
        parts_val=[]; parts_hold=[]; models={}
        for side in ['LONG','SHORT']:
            model=fit_model(train,side); models[side]=model; parts_val.append(add_probs(model,val,side)); parts_hold.append(add_probs(model,hold,side))
        pv=pd.concat(parts_val,ignore_index=True); ph=pd.concat(parts_hold,ignore_index=True)
        for tl in THRESHOLDS:
          for ts in THRESHOLDS:
            tv=simulate(pv,float(tl),float(ts)); mv=metrics(tv); sc=score_val(mv)
            if sc<=-1e8: continue
            all_grid.append({'score':float(sc),'stop_mult':sm,'rr':rr,'tl':float(tl),'ts':float(ts),'shadow_auc_long':shadow_auc['long'],'shadow_auc_short':shadow_auc['short'],'val':mv})
        bundles[(sm,rr)]=(z,pv,ph,shadow_auc)
    if not all_grid: raise RuntimeError('No 2025 validation configuration passed minimum robustness requirements')
    all_grid.sort(key=lambda r:r['score'],reverse=True)
    # require a configuration to be near the top, not a lonely one-off threshold point: score the local threshold neighborhood.
    prelim=all_grid[:60]
    for r in prelim:
        neighbors=[q['score'] for q in all_grid if q['stop_mult']==r['stop_mult'] and q['rr']==r['rr'] and abs(q['tl']-r['tl'])<=.021 and abs(q['ts']-r['ts'])<=.021]
        r['plateau_score']=float(np.median(neighbors)) if neighbors else r['score']
    best=max(prelim,key=lambda r:(r['plateau_score'],r['score']))
    sm,rr,tl,ts=best['stop_mult'],best['rr'],best['tl'],best['ts']; z,pv,ph,shadow_auc=bundles[(sm,rr)]
    # Recompute 2024 shadow strategy with thresholds fixed from 2025, just diagnostic. 2026 remains untouched until here.
    y23=z[z.year==2023]; p24=[]
    for side in ['LONG','SHORT']:
        m=fit_model(y23,side); p24.append(add_probs(m,z[z.year==2024],side))
    p24=pd.concat(p24,ignore_index=True); t24=simulate(p24,tl,ts); t25=simulate(pv,tl,ts); t26=simulate(ph,tl,ts)
    m24,m25,m26=metrics(t24),metrics(t25),metrics(t26)
    hold_pass=bool(m26.get('trades',0)>=25 and m26.get('PF',0)>=1.75 and m26.get('avg_active_day_R',-9)>=1.55 and m26.get('green_pct',0)>=55 and m26['long']['netR']>0 and m26['short']['netR']>0 and m26.get('worst_day_R',-99)>=-3.15)
    strong_pass=bool(hold_pass and m26['PF']>=2.0 and m26['avg_active_day_R']>=1.78 and m26['green_pct']>=60)
    report={'architecture':'V12 ML location-aware router. 5m continuous MGC. Side-specific HGB probability router over broad anti-chase candidate universe.',
      'data':{'bars':int(len(x)),'start':str(x.index.min()),'end':str(x.index.max()),'base_candidates':int(len(base))},
      'frozen_selection':{'train':'2023-2024','threshold_selection':'2025 only','final_holdout':'2026 untouched until final reveal','stop_mult':sm,'rr':rr,'threshold_long':tl,'threshold_short':ts,'plateau_score':best['plateau_score']},
      'risk_controls':{'cost_R':COST_R,'max_trades_day':MAX_TRADES_DAY,'day_stop_R':DAY_STOP_R,'same_direction_loss_kill':DIR_LOSS_KILL,'session_CT':'02:00-15:15','max_hold_bars_5m':HORIZON,'explicit_anti_chase':'short requires loc20>=0.24; long requires loc20<=0.76 plus direction-aware candle/reclaim condition'},
      '2024_shadow_model_trained_2023':m24,'2024_auc':shadow_auc,'2025_validation':m25,'2026_untouched_holdout':m26,'holdout_bootstrap':bootstrap(t26),
      'promotion_pass':hold_pass,'strong_target_pass':strong_pass,
      'promotion_gate':'Holdout: >=25 trades, PF>=1.75, avg active day>=1.55R, green>=55%, both directions positive, worst day>=-3.15R. Strong target: PF>=2, avg>=1.78R (~$400 at $225/R), green>=60%.',
      'warning':'Backtests and ML scores do not guarantee live profitability. Dollar figures are research normalizations only.'}
    OUT.joinpath('V12_ML_ROUTER_REPORT.json').write_text(json.dumps(report,indent=2,default=str))
    t24.to_csv(OUT/'V12_TRADES_2024_SHADOW.csv',index=False); t25.to_csv(OUT/'V12_TRADES_2025_VALIDATION.csv',index=False); t26.to_csv(OUT/'V12_TRADES_2026_HOLDOUT.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='val'}|{f'val_{k}':v for k,v in r['val'].items() if not isinstance(v,dict)} for r in all_grid[:100]]).to_csv(OUT/'V12_VALIDATION_GRID_TOP100.csv',index=False)
    print(json.dumps(report,indent=2,default=str))

if __name__=='__main__': main()
