from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

CT='America/Chicago'
SRC=Path('mgc-source/aggregated/continuous/ohlcv_MGC_5m_continuous.csv')
OUT=Path('mgc_research/stable_results'); OUT.mkdir(parents=True,exist_ok=True)
COST_R=0.05
RESEARCH_RISK_DOLLARS=225.0
SAFE_RISK_DOLLARS=150.0
HORIZON=8
SESSION_START=120
SESSION_LAST_ENTRY=915
MAX_TRADES_DAY=6
DAY_STOP_R=2.15
DIR_LOSS_KILL=2
FAMILY_LOSS_KILL=1

PROFILES=[
    ('P075_R175',0.75,1.75),('P075_R225',0.75,2.25),('P075_R275',0.75,2.75),
    ('P090_R175',0.90,1.75),('P090_R225',0.90,2.25),('P090_R275',0.90,2.75),
    ('P105_R175',1.05,1.75),('P105_R225',1.05,2.25),('P105_R275',1.05,2.75),
]

def load():
    x=pd.read_csv(SRC)
    tcol='timestamp' if 'timestamp' in x.columns else ('datetime' if 'datetime' in x.columns else x.columns[0])
    x['ts']=pd.to_datetime(x[tcol],utc=True,errors='coerce')
    x=x.dropna(subset=['ts']).set_index('ts').sort_index()
    x=x[~x.index.duplicated(keep='last')]
    for c in ['open','high','low','close','volume']:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna(subset=['open','high','low','close','volume'])

def feats(x):
    x=x.copy(); ct=x.index.tz_convert(CT)
    x['date']=ct.date; x['minute']=ct.hour*60+ct.minute; x['year']=ct.year
    o=x.open.astype(float); h=x.high.astype(float); l=x.low.astype(float); c=x.close.astype(float); v=x.volume.astype(float)
    pc=c.shift(); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    x['atr']=tr.ewm(alpha=1/14,adjust=False).mean().clip(lower=.05)
    e9=c.ewm(span=9,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    x['ema9_d']=(c-e9)/(x.atr+1e-9); x['gap20_50']=(e20-e50)/(x.atr+1e-9)
    for n in [12,20,60]:
        rh=h.rolling(n).max(); rl=l.rolling(n).min(); x[f'loc{n}']=(c-rl)/(rh-rl+1e-9)
    x['rh12_prev']=h.shift(1).rolling(12).max(); x['rl12_prev']=l.shift(1).rolling(12).min()
    x['body_atr']=(c-o)/(x.atr+1e-9); x['range_atr']=(h-l)/(x.atr+1e-9); x['close_loc']=(c-l)/(h-l+1e-9)
    x['ret1_atr']=(c-c.shift(1))/(x.atr+1e-9); x['ret3_atr']=(c-c.shift(3))/(x.atr+1e-9); x['ret6_atr']=(c-c.shift(6))/(x.atr+1e-9)
    vm=v.rolling(20).mean(); vs=v.rolling(20).std(); x['volz']=(v-vm)/(vs+1e-9)
    r=c.pct_change(); x['rv_ratio']=r.rolling(6).std()/(r.rolling(24).std()+1e-12)
    typ=(h+l+c)/3; x['vwap']=(typ*v).groupby(x.date).cumsum()/v.groupby(x.date).cumsum().replace(0,np.nan)
    x['vwap_d']=(c-x.vwap)/(x.atr+1e-9)
    x['prev_close']=c.shift(1); x['prev_vwap']=x.vwap.shift(1)
    return x

def session_bucket(m):
    if m<360:return '02_06'
    if m<510:return '06_0830'
    if m<660:return '0830_11'
    if m<810:return '11_1330'
    return '1330_1515'

def trend_bucket(g):
    if g>=.45:return 'SU'
    if g>=.15:return 'UP'
    if g<=-.45:return 'SD'
    if g<=-.15:return 'DN'
    return 'FLAT'

def events(x):
    eligible=(x.minute>=SESSION_START)&(x.minute<=SESSION_LAST_ENTRY)
    good=np.isfinite(x[['atr','loc20','loc60','gap20_50','vwap_d','ret3_atr','ret6_atr','body_atr','close_loc','rh12_prev','rl12_prev','volz','rv_ratio']]).all(axis=1)
    base=eligible&good
    masks={}
    # Exhaustion/reversal entries must already show rejection; do not sell a waterfall low or buy a vertical high.
    masks['EXHAUST_SHORT']=base&(x.loc60>=.80)&(x.vwap_d>=.48)&(x.ret3_atr>=.28)&(x.close_loc<=.62)&(x.body_atr<=.10)
    masks['EXHAUST_LONG']=base&(x.loc60<=.20)&(x.vwap_d<=-.48)&(x.ret3_atr<=-.28)&(x.close_loc>=.38)&(x.body_atr>=-.10)
    # Failed breakout/sweep and reclaim.
    masks['SWEEP_SHORT']=base&(x.high>x.rh12_prev+.04*x.atr)&(x.close<x.rh12_prev)&(x.close_loc<=.58)&(x.body_atr<=.12)
    masks['SWEEP_LONG']=base&(x.low<x.rl12_prev-.04*x.atr)&(x.close>x.rl12_prev)&(x.close_loc>=.42)&(x.body_atr>=-.12)
    # Trend pullback continuation: side-aligned trend but entry bar is not at the chased extreme.
    masks['PULLBACK_LONG']=base&(x.gap20_50>=.22)&(x.loc20.between(.22,.72))&(x.vwap_d>=-.22)&(x.ret3_atr<=.38)&(x.close_loc>=.50)&(x.body_atr>=-.08)
    masks['PULLBACK_SHORT']=base&(x.gap20_50<=-.22)&(x.loc20.between(.28,.78))&(x.vwap_d<=.22)&(x.ret3_atr>=-.38)&(x.close_loc<=.50)&(x.body_atr<=.08)
    # VWAP reclaim continuation after returning through value.
    masks['VWAP_LONG']=base&(x.prev_close<=x.prev_vwap)&(x.close>x.vwap)&(x.gap20_50>=.08)&(x.loc20<=.78)&(x.close_loc>=.55)
    masks['VWAP_SHORT']=base&(x.prev_close>=x.prev_vwap)&(x.close<x.vwap)&(x.gap20_50<=-.08)&(x.loc20>=.22)&(x.close_loc<=.45)
    rows=[]
    for setup,m in masks.items():
        side='LONG' if setup.endswith('LONG') else 'SHORT'
        inds=np.flatnonzero(m.to_numpy())
        for i in inds:
            r=x.iloc[i]
            rows.append((i,setup,side,r.date,int(r.year),int(r.minute),float(r.atr),float(r.gap20_50),float(r.vwap_d),float(r.loc20),float(r.loc60),float(r.ret3_atr),float(r.ret6_atr),float(r.body_atr),float(r.close_loc),float(r.volz),float(r.rv_ratio)))
    e=pd.DataFrame(rows,columns=['i','setup','side','date','year','minute','atr','gap','vwap_d','loc20','loc60','ret3','ret6','body','close_loc','volz','rv_ratio'])
    if len(e)==0:return e
    e['session']=[session_bucket(int(v)) for v in e.minute]
    e['trend']=[trend_bucket(float(v)) for v in e.gap]
    e['family']=e.setup+'|'+e.session+'|'+e.trend
    return e.sort_values(['i','setup']).reset_index(drop=True)

def label_profile(x,e,stop_mult,rr):
    if len(e)==0:return e.copy()
    idx=e.i.to_numpy(dtype=int); ep=idx+1
    valid=ep<len(x); e=e.loc[valid].copy(); idx=e.i.to_numpy(dtype=int); ep=idx+1
    op=x.open.to_numpy(float); hi=x.high.to_numpy(float); lo=x.low.to_numpy(float)
    day=pd.factorize(x.date)[0]
    atr=e.atr.to_numpy(float); side=e.side.to_numpy()
    en=op[ep]; dist=stop_mult*atr
    stop=np.where(side=='LONG',en-dist,en+dist); target=np.where(side=='LONG',en+rr*dist,en-rr*dist)
    offs=np.arange(HORIZON,dtype=int)
    wi=ep[:,None]+offs[None,:]; inrange=wi<len(x); wi=np.minimum(wi,len(x)-1)
    same=inrange&(day[wi]==day[ep][:,None])
    wh=hi[wi]; wl=lo[wi]
    islong=(side=='LONG')[:,None]
    sh=np.where(islong,wl<=stop[:,None],wh>=stop[:,None])&same
    th=np.where(islong,wh>=target[:,None],wl<=target[:,None])&same
    sentinel=HORIZON+1
    fs=np.where(sh.any(axis=1),sh.argmax(axis=1),sentinel)
    ft=np.where(th.any(axis=1),th.argmax(axis=1),sentinel)
    win=ft<fs
    exoff=np.minimum(np.minimum(fs,ft),HORIZON-1)
    e['entry_i']=ep; e['exit_i']=ep+exoff; e['entry']=en; e['stop']=stop; e['target']=target
    e['win']=win.astype(int); e['netR']=np.where(win,rr-COST_R,-1.0-COST_R)
    return e

def pf(s):
    s=np.asarray(s,float); gp=s[s>0].sum(); gl=-s[s<=0].sum(); return float(gp/gl) if gl>0 else 99.0

def family_stats(t,years=(2023,2024)):
    rows=[]
    for fam,g in t[t.year.isin(years)].groupby('family'):
        rec={'family':fam,'setup':g.setup.iloc[0],'side':g.side.iloc[0],'n':len(g),'PF':pf(g.netR),'expR':float(g.netR.mean())}
        ok=True
        for y in years:
            q=g[g.year==y]; rec[f'n{y}']=len(q); rec[f'pf{y}']=pf(q.netR) if len(q) else 0.0; rec[f'net{y}']=float(q.netR.sum()) if len(q) else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)

def select_families(fs,tier):
    if tier=='STRICT':
        m=(fs.n2023>=7)&(fs.n2024>=7)&(fs.PF>=1.50)&(fs.pf2023>=1.10)&(fs.pf2024>=1.10)&(fs.expR>=.12)&(fs.net2023>0)&(fs.net2024>0)
    else:
        m=(fs.n2023>=6)&(fs.n2024>=6)&(fs.PF>=1.38)&(fs.pf2023>=1.00)&(fs.pf2024>=1.00)&(fs.expR>=.09)&(fs.net2023>0)&(fs.net2024>0)
    z=fs[m].copy()
    if len(z): z['rank_score']=z.expR+.15*np.minimum(z.PF,3.0)+.02*np.log1p(z.n)
    return z.sort_values('rank_score',ascending=False) if len(z) else z

def simulate(t,selected,slots):
    if len(selected)==0:return pd.DataFrame()
    score=dict(zip(selected.family,selected.rank_score)); q=t[t.family.isin(score)].copy(); q['qscore']=q.family.map(score)
    q=q.sort_values(['entry_i','qscore'],ascending=[True,False])
    out=[]; active=[]; dayp=defaultdict(float); dayn=defaultdict(int); dl=defaultdict(lambda:defaultdict(int)); fl=defaultdict(set)
    last_entry_key=set()
    for _,r in q.iterrows():
        d=r.date; ei=int(r.entry_i); side=r.side; fam=r.family
        active=[a for a in active if a['exit_i']>=ei]
        if dayn[d]>=MAX_TRADES_DAY or dayp[d]<=-DAY_STOP_R or dl[d][side]>=DIR_LOSS_KILL or fam in fl[d]: continue
        if len(active)>=slots: continue
        if active and any(a['side']!=side for a in active): continue
        key=(ei,fam)
        if key in last_entry_key: continue
        last_entry_key.add(key)
        nr=float(r.netR)
        out.append({'date':d,'year':int(r.year),'setup':r.setup,'family':fam,'side':side,'entry_i':ei,'exit_i':int(r.exit_i),'netR':nr,'entry':float(r.entry),'stop':float(r.stop),'target':float(r.target)})
        dayp[d]+=nr; dayn[d]+=1
        if nr<0: dl[d][side]+=1; fl[d].add(fam)
        active.append({'exit_i':int(r.exit_i),'side':side})
    return pd.DataFrame(out)

def metrics(t):
    if t is None or len(t)==0:return {'trades':0}
    s=t.netR.to_numpy(float); daily=t.groupby('date').netR.sum(); all_dates=len(daily)
    res={'trades':int(len(t)),'active_days':int(all_dates),'tpd':float(len(t)/all_dates),'netR':float(s.sum()),'PF':pf(s),'win_pct':float((s>0).mean()*100),'green_pct':float((daily>0).mean()*100),'avg_day_R':float(daily.mean()),'worst_day_R':float(daily.min()),'best_day_R':float(daily.max())}
    for side in ['LONG','SHORT']:
        z=t[t.side==side]; res[side.lower()]={'trades':int(len(z)),'netR':float(z.netR.sum()) if len(z) else 0.0,'PF':pf(z.netR) if len(z) else 0.0}
    res['research_dollars']={'avg_day':res['avg_day_R']*RESEARCH_RISK_DOLLARS,'worst_day':res['worst_day_R']*RESEARCH_RISK_DOLLARS,'net':res['netR']*RESEARCH_RISK_DOLLARS}
    res['safe_dollars']={'avg_day':res['avg_day_R']*SAFE_RISK_DOLLARS,'worst_day':res['worst_day_R']*SAFE_RISK_DOLLARS,'net':res['netR']*SAFE_RISK_DOLLARS}
    return res

def score_pre(m23,m24):
    if m23.get('trades',0)<40 or m24.get('trades',0)<40:return -1e9
    if m23['long']['netR']<=0 or m23['short']['netR']<=0 or m24['long']['netR']<=0 or m24['short']['netR']<=0:return -1e9
    return min(m23['avg_day_R'],m24['avg_day_R'])+.35*min(m23['PF'],m24['PF'],3.0)+.004*min(m23['green_pct'],m24['green_pct'])-.10*max(0,abs(min(m23['worst_day_R'],m24['worst_day_R']))-3.1)

def main():
    x=feats(load()); e=events(x)
    summary=[]; labeled={}
    for name,sm,rr in PROFILES:
        t=label_profile(x,e,sm,rr); labeled[name]=t
        fs=family_stats(t)
        for tier in ['STRICT','BALANCED']:
            sel=select_families(fs,tier)
            for slots in [1,2]:
                p23=simulate(t[t.year==2023],sel,slots); p24=simulate(t[t.year==2024],sel,slots)
                m23=metrics(p23); m24=metrics(p24); sc=score_pre(m23,m24)
                summary.append({'profile':name,'stop_mult':sm,'rr':rr,'tier':tier,'slots':slots,'families':int(len(sel)),'pre_score':sc,'y2023':m23,'y2024':m24})
    valid=[r for r in summary if r['pre_score']>-1e8]
    if not valid: raise RuntimeError('No architecture passed pre-2025 two-year stability requirements')
    valid.sort(key=lambda r:r['pre_score'],reverse=True); best=valid[0]
    t=labeled[best['profile']]; fs=family_stats(t); sel=select_families(fs,best['tier']); slots=best['slots']
    tr23=simulate(t[t.year==2023],sel,slots); tr24=simulate(t[t.year==2024],sel,slots); tr25=simulate(t[t.year==2025],sel,slots); tr26=simulate(t[t.year==2026],sel,slots)
    m23,m24,m25,m26=map(metrics,[tr23,tr24,tr25,tr26])
    verification_pass=bool(m25.get('trades',0)>=35 and m25.get('PF',0)>=1.45 and m25.get('avg_day_R',-9)>=1.0 and m25['long']['netR']>0 and m25['short']['netR']>0 and m25.get('worst_day_R',-99)>=-3.15)
    holdout_pass=bool(m26.get('trades',0)>=20 and m26.get('PF',0)>=1.75 and m26.get('avg_day_R',-9)>=1.78 and m26.get('green_pct',0)>=55 and m26['long']['netR']>0 and m26['short']['netR']>0 and m26.get('worst_day_R',-99)>=-3.15)
    report={
      'architecture':'V11 stable-family router: exhaustion/sweep/pullback/VWAP, selected only from 2023-2024; 2025 verification; 2026 untouched holdout',
      'data':{'bars':int(len(x)),'start':str(x.index.min()),'end':str(x.index.max()),'events':int(len(e))},
      'frozen_pre2025_selection':{k:best[k] for k in ['profile','stop_mult','rr','tier','slots','families','pre_score']},
      'risk_controls':{'cost_R':COST_R,'max_trades_day':MAX_TRADES_DAY,'day_stop_R':DAY_STOP_R,'direction_loss_kill':DIR_LOSS_KILL,'family_loss_kill':FAMILY_LOSS_KILL,'session_CT':'02:00-15:15 entries; max 40m hold'},
      'selected_families':sel.to_dict('records'),
      '2023_development':m23,'2024_development':m24,'2025_unseen_verification':m25,'2026_untouched_holdout':m26,
      'verification_pass':verification_pass,'holdout_promotion_pass':holdout_pass,'overall_promotion_pass':bool(verification_pass and holdout_pass),
      'promotion_gate':'2025 must remain robust; 2026 requires PF>=1.75, avg active day >=1.78R (~$400 at $225/R), green>=55%, both directions positive, worst day >=-3.15R.',
      'warning':'Backtests do not guarantee live profit. Research dollars are a normalization, not a promise or recommended current-account risk.'
    }
    OUT.joinpath('V11_STABLE_FAMILY_REPORT.json').write_text(json.dumps(report,indent=2,default=str))
    sel.to_csv(OUT/'V11_SELECTED_FAMILIES.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['y2023','y2024']}|{'y2023_PF':r['y2023'].get('PF'),'y2023_avgR':r['y2023'].get('avg_day_R'),'y2024_PF':r['y2024'].get('PF'),'y2024_avgR':r['y2024'].get('avg_day_R')} for r in valid[:50]]).to_csv(OUT/'V11_PRE2025_GRID.csv',index=False)
    for y,tr in [(2023,tr23),(2024,tr24),(2025,tr25),(2026,tr26)]: tr.to_csv(OUT/f'V11_TRADES_{y}.csv',index=False)
    print(json.dumps(report,indent=2,default=str))

if __name__=='__main__':main()
