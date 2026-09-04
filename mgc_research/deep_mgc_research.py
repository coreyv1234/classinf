from __future__ import annotations
import json, math, os, subprocess, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CT='America/Chicago'
OUT=Path('mgc_research/results')
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR=Path('mgc-source/data')
COST_R=0.06
SESSION_START=120   # 02:00 CT
SESSION_END=945     # 15:45 CT
MAX_TRADES_DAY=5
DAY_STOP_R=2.15
SAME_DIR_LOSS_KILL=2
WAIT_MINUTES=10
HORIZON_MINUTES=42
SETUPS=['REV_LONG','REV_SHORT','CONT_LONG','CONT_SHORT']

FEATURES=[
 'offset','side_sign','setup_id',
 'm1_body_atr','m1_range_atr','m1_close_loc','m1_ret3_atr','m1_ret5_atr',
 'm1_loc10','m1_loc20','m1_ema9_d','m1_ema21_d','m1_vwap_d','m1_volz20','m1_rv_ratio','m1_er10',
 'sig_gap20_50','sig_loc20','sig_loc60','sig_vwap_d','sig_ret3_atr','sig_ret6_atr','sig_body_atr','sig_close_loc','sig_volz20','sig_rv_ratio',
 'from_sig_atr','from_extreme_atr','retracement_atr','sin_t','cos_t'
]

def load_data():
    files=sorted(DATA_DIR.glob('ohlcv_MGC_*.csv'))
    if not files: raise RuntimeError('No MGC daily CSV files found')
    frames=[]
    for f in files:
        try:
            d=pd.read_csv(f)
            if len(d)<50: continue
            frames.append(d)
        except Exception:
            continue
    x=pd.concat(frames, ignore_index=True)
    x['ts']=pd.to_datetime(x['timestamp'], utc=True, errors='coerce')
    x=x.dropna(subset=['ts']).set_index('ts').sort_index()
    x=x[~x.index.duplicated(keep='last')]
    for c in ['open','high','low','close','volume']:
        x[c]=pd.to_numeric(x[c], errors='coerce')
    x=x.dropna(subset=['open','high','low','close','volume'])
    return x

def add_1m_features(x):
    x=x.copy()
    ct=x.index.tz_convert(CT)
    x['date_ct']=ct.date
    x['min_ct']=ct.hour*60+ct.minute
    o,h,l,c,v=[x[k].astype(float) for k in ['open','high','low','close','volume']]
    pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    x['atr1']=tr.ewm(alpha=1/20,adjust=False).mean().clip(lower=.05)
    for n in [9,21]:
        ema=c.ewm(span=n,adjust=False).mean(); x[f'ema{n}_d']=(c-ema)/(x.atr1+1e-9)
    for n in [10,20]:
        hi=h.rolling(n).max(); lo=l.rolling(n).min(); x[f'loc{n}']=(c-lo)/(hi-lo+1e-9)
    x['body_atr']=(c-o)/(x.atr1+1e-9)
    x['range_atr']=(h-l)/(x.atr1+1e-9)
    x['close_loc']=(c-l)/(h-l+1e-9)
    x['ret3_atr']=(c-c.shift(3))/(x.atr1+1e-9)
    x['ret5_atr']=(c-c.shift(5))/(x.atr1+1e-9)
    vm=v.rolling(20).mean(); vs=v.rolling(20).std(); x['volz20']=(v-vm)/(vs+1e-9)
    r=c.pct_change(); rv10=r.rolling(10).std(); rv60=r.rolling(60).std(); x['rv_ratio']=rv10/(rv60+1e-12)
    x['er10']=(c-c.shift(10)).abs()/(c.diff().abs().rolling(10).sum()+1e-9)
    typ=(h+l+c)/3
    x['vwap']=(typ*v).groupby(x.date_ct).cumsum()/v.groupby(x.date_ct).cumsum().replace(0,np.nan)
    x['vwap_d']=(c-x.vwap)/(x.atr1+1e-9)
    return x

def make_5m(x):
    a=x[['open','high','low','close','volume']].resample('5min',closed='left',label='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    ct=a.index.tz_convert(CT); a['date_ct']=ct.date; a['min_ct']=ct.hour*60+ct.minute
    o,h,l,c,v=[a[k].astype(float) for k in ['open','high','low','close','volume']]
    pc=c.shift(); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    a['atr']=tr.ewm(alpha=1/14,adjust=False).mean().clip(lower=.1)
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    a['gap20_50']=(e20-e50)/(a.atr+1e-9)
    for n in [20,60]:
        hi=h.rolling(n).max(); lo=l.rolling(n).min(); a[f'loc{n}']=(c-lo)/(hi-lo+1e-9)
    a['body_atr']=(c-o)/(a.atr+1e-9); a['close_loc']=(c-l)/(h-l+1e-9)
    a['ret3_atr']=(c-c.shift(3))/(a.atr+1e-9); a['ret6_atr']=(c-c.shift(6))/(a.atr+1e-9)
    vm=v.rolling(20).mean(); vs=v.rolling(20).std(); a['volz20']=(v-vm)/(vs+1e-9)
    r=c.pct_change(); a['rv_ratio']=r.rolling(5).std()/(r.rolling(20).std()+1e-12)
    typ=(h+l+c)/3; a['vwap']=(typ*v).groupby(a.date_ct).cumsum()/v.groupby(a.date_ct).cumsum().replace(0,np.nan)
    a['vwap_d']=(c-a.vwap)/(a.atr+1e-9)
    return a

def detect_opps(a):
    rows=[]
    for i in range(80,len(a)-20):
        r=a.iloc[i]; m=int(r.min_ct)
        if m<SESSION_START or m>SESSION_END: continue
        vals=[r.atr,r.gap20_50,r.loc20,r.loc60,r.vwap_d,r.ret3_atr,r.ret6_atr,r.body_atr,r.close_loc,r.volz20,r.rv_ratio]
        if not np.all(np.isfinite(vals)): continue
        setup=None; side=None
        # reversal opportunities demand location + displacement, not merely a candle color
        if r.loc60>=.80 and r.vwap_d>=.45 and r.ret3_atr>=.30:
            setup='REV_SHORT'; side='SHORT'
        elif r.loc60<=.20 and r.vwap_d<=-.45 and r.ret3_atr<=-.30:
            setup='REV_LONG'; side='LONG'
        # continuation opportunities are trend aligned and require a pullback away from an extreme chase
        elif r.gap20_50>=.24 and r.loc20<=.72 and r.vwap_d>=-.15 and r.ret6_atr>=.20:
            setup='CONT_LONG'; side='LONG'
        elif r.gap20_50<=-.24 and r.loc20>=.28 and r.vwap_d<=.15 and r.ret6_atr<=-.20:
            setup='CONT_SHORT'; side='SHORT'
        if setup:
            rows.append({'sig_ts':a.index[i],'setup':setup,'side':side,'sig_atr':float(r.atr),'sig_close':float(r.close),
                         'sig_gap20_50':float(r.gap20_50),'sig_loc20':float(r.loc20),'sig_loc60':float(r.loc60),
                         'sig_vwap_d':float(r.vwap_d),'sig_ret3_atr':float(r.ret3_atr),'sig_ret6_atr':float(r.ret6_atr),
                         'sig_body_atr':float(r.body_atr),'sig_close_loc':float(r.close_loc),'sig_volz20':float(r.volz20),'sig_rv_ratio':float(r.rv_ratio)})
    return pd.DataFrame(rows)

def barrier(x, entry_pos, side, stop_dist, rr):
    if entry_pos>=len(x): return None
    en=float(x.open.iloc[entry_pos]); d=x.date_ct.iloc[entry_pos]
    if side=='LONG': stop=en-stop_dist; target=en+rr*stop_dist
    else: stop=en+stop_dist; target=en-rr*stop_dist
    end=min(len(x)-1,entry_pos+HORIZON_MINUTES)
    for j in range(entry_pos,end+1):
        if x.date_ct.iloc[j]!=d: return (0,j-1,en,stop,target)
        hi=float(x.high.iloc[j]); lo=float(x.low.iloc[j])
        # Conservative if stop and target are both touched in one 1m bar: stop wins.
        if side=='LONG':
            if lo<=stop: return (0,j,en,stop,target)
            if hi>=target: return (1,j,en,stop,target)
        else:
            if hi>=stop: return (0,j,en,stop,target)
            if lo<=target: return (1,j,en,stop,target)
    return (0,end,en,stop,target)

def candidate_table(x, opps):
    pos_map=pd.Series(np.arange(len(x)),index=x.index)
    setup_id={s:i for i,s in enumerate(SETUPS)}
    rec=[]
    for oid,o in opps.iterrows():
        if o.sig_ts not in pos_map.index: continue
        start=int(pos_map.loc[o.sig_ts])
        atr=float(o.sig_atr)
        if atr<=0: continue
        side=o.side; sign=1.0 if side=='LONG' else -1.0
        # Candidate bar closes from signal close through +9m; entry is next 1m open.
        sig_close=float(o.sig_close)
        local=x.iloc[max(0,start-20):start+WAIT_MINUTES+1]
        pre_hi=float(local.high.iloc[:min(21,len(local))].max()); pre_lo=float(local.low.iloc[:min(21,len(local))].min())
        for off in range(WAIT_MINUTES):
            ci=start+off
            ep=ci+1
            if ep>=len(x) or x.date_ct.iloc[ep]!=x.date_ct.iloc[start]: break
            r=x.iloc[ci]
            need=['body_atr','range_atr','close_loc','ret3_atr','ret5_atr','loc10','loc20','ema9_d','ema21_d','vwap_d','volz20','rv_ratio','er10']
            if not np.all(np.isfinite([r[k] for k in need])): continue
            # Avoid mechanically chasing after the move has already left the signal.
            from_sig=(float(r.close)-sig_close)/atr
            if side=='SHORT' and from_sig<=-1.25: continue
            if side=='LONG' and from_sig>=1.25: continue
            if side=='SHORT':
                ext=(float(r.high)-pre_lo)/atr
                retr=max(0.0,(float(r.high)-sig_close)/atr)
            else:
                ext=(pre_hi-float(r.low))/atr
                retr=max(0.0,(sig_close-float(r.low))/atr)
            ct=x.index[ci].tz_convert(CT); minute=ct.hour*60+ct.minute
            base={'opp_id':int(oid),'cand_ts':x.index[ci],'entry_pos':ep,'date':x.date_ct.iloc[ci],
                  'year':int(x.index[ci].year),'setup':o.setup,'side':side,'offset':float(off),'side_sign':sign,'setup_id':float(setup_id[o.setup]),
                  'm1_body_atr':float(r.body_atr),'m1_range_atr':float(r.range_atr),'m1_close_loc':float(r.close_loc),'m1_ret3_atr':float(r.ret3_atr),'m1_ret5_atr':float(r.ret5_atr),
                  'm1_loc10':float(r.loc10),'m1_loc20':float(r.loc20),'m1_ema9_d':float(r.ema9_d),'m1_ema21_d':float(r.ema21_d),'m1_vwap_d':float(r.vwap_d),
                  'm1_volz20':float(r.volz20),'m1_rv_ratio':float(r.rv_ratio),'m1_er10':float(r.er10),
                  'from_sig_atr':float(from_sig),'from_extreme_atr':float(ext),'retracement_atr':float(retr),
                  'sin_t':float(np.sin(2*np.pi*minute/1440.0)),'cos_t':float(np.cos(2*np.pi*minute/1440.0))}
            for k in ['sig_gap20_50','sig_loc20','sig_loc60','sig_vwap_d','sig_ret3_atr','sig_ret6_atr','sig_body_atr','sig_close_loc','sig_volz20','sig_rv_ratio']:
                base[k]=float(o[k])
            # stop is volatility based but bounded to avoid tiny noise stops
            stop_dist=max(0.55,0.78*atr)
            for rr in [1.5,2.0,2.5,3.0]:
                b=barrier(x,ep,side,stop_dist,rr)
                if b is None: continue
                y,exit_pos,en,st,tp=b
                base[f'y_{rr}']=int(y); base[f'exit_{rr}']=int(exit_pos); base[f'entry_{rr}']=float(en); base[f'stop_{rr}']=float(st); base[f'target_{rr}']=float(tp)
            rec.append(base)
    return pd.DataFrame(rec)

def model_factory(kind):
    if kind=='HGB':
        return HistGradientBoostingClassifier(max_iter=180,learning_rate=.04,max_depth=3,min_samples_leaf=70,l2_regularization=5.0,random_state=27)
    return make_pipeline(StandardScaler(),LogisticRegression(C=.35,max_iter=1000,class_weight='balanced',random_state=27))

def fit_predict(cands,rr,kind):
    y=f'y_{rr}'
    train=cands[cands.year<=2024].copy(); val=cands[cands.year==2025].copy(); hold=cands[cands.year>=2026].copy()
    model=model_factory(kind); model.fit(train[FEATURES],train[y])
    preds={}
    for name,d in [('train',train),('val',val),('hold',hold)]:
        p=model.predict_proba(d[FEATURES])[:,1]
        preds[name]=pd.Series(p,index=d.index)
    auc={}
    for name,d in [('train',train),('val',val),('hold',hold)]:
        if d[y].nunique()>1: auc[name]=float(roc_auc_score(d[y],preds[name]))
        else: auc[name]=None
    return model,train,val,hold,preds,auc

def simulate(d,p,rr,th_long,th_short):
    # First qualifying candidate after each opportunity; then one-position scheduler.
    z=d.copy(); z['p']=p.reindex(z.index)
    z=z.sort_values(['cand_ts','offset'])
    chosen=[]
    for oid,g in z.groupby('opp_id',sort=False):
        side=g.side.iloc[0]; th=th_long if side=='LONG' else th_short
        q=g[g.p>=th]
        if len(q): chosen.append(q.iloc[0])
    if not chosen: return pd.DataFrame()
    q=pd.DataFrame(chosen).sort_values('cand_ts')
    out=[]; active_until=-1; day_p=defaultdict(float); day_n=defaultdict(int); dir_losses=defaultdict(lambda:defaultdict(int))
    for _,r in q.iterrows():
        ep=int(r.entry_pos); dte=r.date; side=r.side
        if ep<=active_until: continue
        if day_n[dte]>=MAX_TRADES_DAY or day_p[dte]<=-DAY_STOP_R or dir_losses[dte][side]>=SAME_DIR_LOSS_KILL: continue
        win=int(r[f'y_{rr}']); gross=rr if win else -1.0; net=gross-COST_R
        exitp=int(r[f'exit_{rr}'])
        out.append({'cand_ts':r.cand_ts,'date':dte,'year':int(r.year),'setup':r.setup,'side':side,'prob':float(r.p),
                    'entry':float(r[f'entry_{rr}']),'stop':float(r[f'stop_{rr}']),'target':float(r[f'target_{rr}']),
                    'win':win,'netR':float(net),'entry_pos':ep,'exit_pos':exitp})
        day_p[dte]+=net; day_n[dte]+=1
        if net<0: dir_losses[dte][side]+=1
        active_until=exitp
    return pd.DataFrame(out)

def metrics(t):
    if t is None or len(t)==0: return {'trades':0}
    s=t.netR.astype(float); daily=t.groupby('date').netR.sum(); gp=s[s>0].sum(); gl=-s[s<=0].sum()
    def side_stats(side):
        z=t[t.side==side]; a=z.netR.astype(float) if len(z) else pd.Series(dtype=float); loss=-a[a<=0].sum() if len(a) else 0
        return {'trades':int(len(z)),'netR':float(a.sum()) if len(a) else 0.0,'PF':float(a[a>0].sum()/loss) if loss>0 else None}
    return {'trades':int(len(t)),'days':int(len(daily)),'trades_per_active_day':float(len(t)/len(daily)),'netR':float(s.sum()),
            'PF':float(gp/gl) if gl>0 else None,'win_rate_pct':float((s>0).mean()*100),'green_day_pct':float((daily>0).mean()*100),
            'avg_active_day_R':float(daily.mean()),'worst_day_R':float(daily.min()),'best_day_R':float(daily.max()),
            'long':side_stats('LONG'),'short':side_stats('SHORT')}

def monthly(t):
    if len(t)==0: return []
    x=t.copy(); x['month']=pd.to_datetime(x.cand_ts,utc=True).dt.to_period('M').astype(str)
    return [{'month':m,**metrics(g)} for m,g in x.groupby('month')]

def bootstrap(t,n=3000,seed=27):
    if len(t)==0: return {}
    d=t.groupby('date').netR.sum().to_numpy(); rng=np.random.default_rng(seed)
    means=[]; dds=[]
    for _ in range(n):
        s=rng.choice(d,size=len(d),replace=True); means.append(float(np.mean(s)))
        eq=np.cumsum(s); peak=np.maximum.accumulate(np.r_[0,eq]); dd=peak[1:]-eq; dds.append(float(np.max(dd)) if len(dd) else 0)
    return {'avg_day_R_p05':float(np.quantile(means,.05)),'avg_day_R_median':float(np.median(means)),'avg_day_R_p95':float(np.quantile(means,.95)),
            'maxDD_R_median':float(np.median(dds)),'maxDD_R_p95':float(np.quantile(dds,.95))}

def main():
    x=add_1m_features(load_data()); a=make_5m(x); opps=detect_opps(a); cands=candidate_table(x,opps)
    cands.to_parquet(OUT/'candidate_sample.parquet',index=False)
    search=[]; fitted={}
    for rr in [1.5,2.0,2.5,3.0]:
      for kind in ['LOGIT','HGB']:
        model,tr,va,ho,preds,auc=fit_predict(cands,rr,kind)
        fitted[(rr,kind)]=(model,tr,va,ho,preds,auc)
        for tl in np.arange(.50,.76,.025):
          for ts in np.arange(.50,.76,.025):
            tv=simulate(va,preds['val'],rr,float(tl),float(ts)); mv=metrics(tv)
            if mv.get('trades',0)<120: continue
            if mv['long']['trades']<30 or mv['short']['trades']<30: continue
            if mv['long']['netR']<=0 or mv['short']['netR']<=0: continue
            pf=mv['PF'] or 0
            score=mv['avg_active_day_R'] + .30*min(pf,3.0) + .004*mv['green_day_pct'] - .08*max(0,abs(mv['worst_day_R'])-2.2)
            search.append({'score':float(score),'rr':rr,'kind':kind,'tl':float(tl),'ts':float(ts),'val':mv,'auc':auc})
    if not search: raise RuntimeError('No validation configuration met minimum robustness constraints')
    search=sorted(search,key=lambda z:z['score'],reverse=True)
    best=search[0]; rr=best['rr']; kind=best['kind']; tl=best['tl']; ts=best['ts']
    model,tr,va,ho,preds,auc=fitted[(rr,kind)]
    tt=simulate(tr,preds['train'],rr,tl,ts); tv=simulate(va,preds['val'],rr,tl,ts); th=simulate(ho,preds['hold'],rr,tl,ts)
    report={
      'data':{'bars_1m':int(len(x)),'first':str(x.index.min()),'last':str(x.index.max()),'opportunities':int(len(opps)),'candidates':int(len(cands))},
      'frozen_selection':{'selected_on':'2025 validation only','rr':rr,'model':kind,'threshold_long':tl,'threshold_short':ts,'cost_R_per_trade':COST_R,
                          'max_trades_day':MAX_TRADES_DAY,'day_stop_R':DAY_STOP_R,'same_direction_loss_kill':SAME_DIR_LOSS_KILL},
      'auc':auc,'train_2023_2024':metrics(tt),'validation_2025':metrics(tv),'holdout_2026_to_2026_03_23':metrics(th),
      'holdout_monthly':monthly(th),'holdout_bootstrap':bootstrap(th),
      'top_validation_configs':search[:20],
      'promotion_gate':{'PF_min':2.0,'avg_day_R_min':1.8,'green_days_min_pct':60.0,'both_directions_positive':True},
      'warning':'No backtest guarantees live profits. Holdout is untouched by threshold/model selection in this run.'
    }
    hm=report['holdout_2026_to_2026_03_23']
    report['promotion_pass']=bool(hm.get('PF') and hm['PF']>=2.0 and hm['avg_active_day_R']>=1.8 and hm['green_day_pct']>=60 and hm['long']['netR']>0 and hm['short']['netR']>0)
    (OUT/'V11_MULTIYEAR_REPORT.json').write_text(json.dumps(report,indent=2,default=str))
    tt.to_csv(OUT/'train_trades.csv',index=False); tv.to_csv(OUT/'validation_trades.csv',index=False); th.to_csv(OUT/'holdout_trades.csv',index=False)
    pd.DataFrame([{**{k:v for k,v in z.items() if k not in ['val','auc']}, **{f'val_{k}':vv for k,vv in z['val'].items() if not isinstance(vv,dict)}} for z in search]).to_csv(OUT/'validation_grid.csv',index=False)
    print(json.dumps(report,indent=2,default=str))

if __name__=='__main__': main()
