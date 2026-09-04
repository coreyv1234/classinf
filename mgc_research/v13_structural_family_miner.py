from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

CT='America/Chicago'
SRC=Path('mgc-source/aggregated/continuous/ohlcv_MGC_5m_continuous.csv')
OUT=Path('mgc_research/v13_results'); OUT.mkdir(parents=True,exist_ok=True)
COST_R=.06
HOLD=12
MAX_DAY=5
DAYSTOP=-2.20
DIR_LOSS_KILL=2
RISK=225.0

def load():
    x=pd.read_csv(SRC); tc='timestamp' if 'timestamp' in x.columns else x.columns[0]
    x['ts']=pd.to_datetime(x[tc],utc=True,errors='coerce'); x=x.dropna(subset=['ts']).set_index('ts').sort_index(); x=x[~x.index.duplicated(keep='last')]
    for c in ['open','high','low','close','volume']:x[c]=pd.to_numeric(x[c],errors='coerce')
    return x.dropna(subset=['open','high','low','close','volume'])

def adx(h,l,c,n=14):
    up=h.diff(); dn=-l.diff(); plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    pc=c.shift(); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); atr=tr.ewm(alpha=1/n,adjust=False).mean()
    p=100*pd.Series(plus,index=c.index).ewm(alpha=1/n,adjust=False).mean()/(atr+1e-9); m=100*pd.Series(minus,index=c.index).ewm(alpha=1/n,adjust=False).mean()/(atr+1e-9)
    dx=100*(p-m).abs()/(p+m+1e-9); return dx.ewm(alpha=1/n,adjust=False).mean()

def features(x):
    x=x.copy(); ct=x.index.tz_convert(CT); x['date']=ct.date; x['year']=ct.year; x['hour']=ct.hour; x['minute']=ct.hour*60+ct.minute
    o,h,l,c,v=[x[k].astype(float) for k in ['open','high','low','close','volume']]
    pc=c.shift(); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); x['atr']=tr.ewm(alpha=1/14,adjust=False).mean().clip(lower=.05)
    e9=c.ewm(span=9,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    x['ema9']=e9;x['ema20']=e20;x['gap']=(e20-e50)/(x.atr+1e-9);x['e9d']=(c-e9)/(x.atr+1e-9);x['e20d']=(c-e20)/(x.atr+1e-9);x['slope20']=(e20-e20.shift(3))/(x.atr+1e-9)
    for n in [6,12,20,60]:
        x[f'ph{n}']=h.shift(1).rolling(n).max();x[f'pl{n}']=l.shift(1).rolling(n).min(); rh=h.rolling(n).max();rl=l.rolling(n).min();x[f'loc{n}']=(c-rl)/(rh-rl+1e-9)
    x['body']=(c-o)/(x.atr+1e-9);x['range']=(h-l)/(x.atr+1e-9);x['cloc']=(c-l)/(h-l+1e-9)
    x['ret1']=(c-c.shift(1))/(x.atr+1e-9);x['ret3']=(c-c.shift(3))/(x.atr+1e-9);x['ret6']=(c-c.shift(6))/(x.atr+1e-9)
    x['extension']=(c-e20)/(x.atr+1e-9)
    vm=v.rolling(20).mean();x['vr20']=v/(vm+1e-9)
    rz=x['range'];x['rangez']=(rz-rz.rolling(60).mean())/(rz.rolling(60).std()+1e-9)
    mom=x['ret3'];x['momz']=(mom-mom.rolling(60).mean())/(mom.rolling(60).std()+1e-9)
    rv=c.pct_change().rolling(12).std();x['rvrel']=rv/(rv.rolling(100).mean()+1e-12)
    typ=(h+l+c)/3;x['vwap']=(typ*v).groupby(x.date).cumsum()/v.groupby(x.date).cumsum().replace(0,np.nan);x['vwapd']=(c-x.vwap)/(x.atr+1e-9)
    x['adx']=adx(h,l,c)
    x['run3']=np.sign(c.diff()).rolling(3).sum()
    x['run3_prev']=x.run3.shift(1)
    return x

def outcome(x, idx, side, stop_price, rr):
    idx=np.asarray(idx,int); ep=idx+1; keep=ep<len(x);idx=idx[keep];ep=ep[keep];stop_price=np.asarray(stop_price,float)[keep]
    if len(idx)==0:return pd.DataFrame()
    op=x.open.to_numpy(float);hi=x.high.to_numpy(float);lo=x.low.to_numpy(float);cl=x.close.to_numpy(float);dates=np.asarray(x.date);years=np.asarray(x.year)
    en=op[ep]; dist=np.abs(en-stop_price); good=np.isfinite(dist)&(dist>=.20)&(dist<=40.0)
    idx=idx[good];ep=ep[good];stop_price=stop_price[good];en=en[good];dist=dist[good]
    target=np.where(side=='LONG',en+rr*dist,en-rr*dist)
    offs=np.arange(HOLD); wi=ep[:,None]+offs[None,:];valid=wi<len(x);wi=np.minimum(wi,len(x)-1);same=valid&(dates[wi]==dates[ep][:,None]);hh=hi[wi];ll=lo[wi]
    if side=='LONG':sh=(ll<=stop_price[:,None])&same;th=(hh>=target[:,None])&same
    else:sh=(hh>=stop_price[:,None])&same;th=(ll<=target[:,None])&same
    sent=HOLD+1;fs=np.where(sh.any(1),sh.argmax(1),sent);ft=np.where(th.any(1),th.argmax(1),sent);win=ft<fs; stopped=fs<=ft
    first=np.minimum(fs,ft); timeout=first>HOLD-1; exoff=np.where(timeout,HOLD-1,first); exi=ep+exoff
    mtm=np.where(side=='LONG',(cl[exi]-en)/dist,(en-cl[exi])/dist); gross=np.where(win,rr,np.where(stopped,-1.0,np.clip(mtm,-1.0,rr))); net=gross-COST_R
    return pd.DataFrame({'signal_i':idx,'entry_i':ep,'exit_i':exi,'signal_ts':x.index[idx],'date':dates[idx],'year':years[idx],'side':side,'entry':en,'stop':stop_price,'target':target,'netR':net,'win':net>0})

def metric(t):
    if len(t)==0:return {'n':0,'PF':0,'expR':-9,'win':0}
    s=t.netR.to_numpy(float);gp=s[s>0].sum();gl=-s[s<=0].sum();return {'n':int(len(t)),'PF':float(gp/gl) if gl>0 else 99.0,'expR':float(s.mean()),'win':float((s>0).mean()*100),'netR':float(s.sum())}

def add_specs(x):
    specs=[]
    session=(x.minute>=120)&(x.minute<=915)
    finite=np.isfinite(x[['atr','vwap','body','cloc','extension','rangez','momz','vr20','gap','adx','ph12','pl12']]).all(axis=1)
    base=session&finite
    # Failed sweeps: structural stop beyond the rejection bar. Mirrors are independent families.
    for side in ['SHORT','LONG']:
      for buf in [.00,.03,.06]:
       for cloc in [.40,.48,.56]:
        for ext in [.35,.65,1.0]:
         for rr in [.8,1.0,1.2,1.5]:
          if side=='SHORT':m=base&(x.high>x.ph12+buf*x.atr)&(x.close<x.ph12)&(x.body<-.01)&(x.cloc<=cloc)&(x.extension>=ext)
          else:m=base&(x.low<x.pl12-buf*x.atr)&(x.close>x.pl12)&(x.body>.01)&(x.cloc>=1-cloc)&(x.extension<=-ext)
          idx=np.flatnonzero(m.to_numpy());stop=(x.high.iloc[idx]+.06*x.atr.iloc[idx]).to_numpy() if side=='SHORT' else (x.low.iloc[idx]-.06*x.atr.iloc[idx]).to_numpy()
          specs.append((f'SWEEP_{side}_b{buf}_c{cloc}_e{ext}_r{rr}',side,rr,idx,stop))
    # Exhaustion: extended move followed by bar breaking previous opposite extreme.
    for side in ['SHORT','LONG']:
      for ext in [.75,1.0,1.25]:
       for body in [.02,.05,.09]:
        for vol in [.75,.95,1.15]:
         for rr in [1.0,1.2,1.5]:
          if side=='SHORT':m=base&(x.extension.shift(1)>=ext)&(x.run3_prev>=1)&(x.body<=-body)&(x.cloc<=.48)&(x.close<x.low.shift(1))&(x.vr20>=vol)
          else:m=base&(x.extension.shift(1)<=-ext)&(x.run3_prev<=-1)&(x.body>=body)&(x.cloc>=.52)&(x.close>x.high.shift(1))&(x.vr20>=vol)
          idx=np.flatnonzero(m.fillna(False).to_numpy());stop=(pd.concat([x.high,x.high.shift(1)],axis=1).max(axis=1).iloc[idx]+.06*x.atr.iloc[idx]).to_numpy() if side=='SHORT' else (pd.concat([x.low,x.low.shift(1)],axis=1).min(axis=1).iloc[idx]-.06*x.atr.iloc[idx]).to_numpy()
          specs.append((f'EXHAUST_{side}_e{ext}_b{body}_v{vol}_r{rr}',side,rr,idx,stop))
    # VWAP rejection/reclaim. Requires actual completed cross plus directional candle.
    for side in ['SHORT','LONG']:
      for body in [.02,.05,.10]:
       for hour in [2,6,8]:
        for trend in [-.15,.0,.15]:
         for rr in [.8,1.0,1.25,1.5]:
          if side=='SHORT':m=base&(x.hour>=hour)&(x.close.shift(1)>=x.vwap.shift(1))&(x.close<x.vwap)&(x.body<=-body)&(x.cloc<=.52)&(x.gap>=trend)
          else:m=base&(x.hour>=hour)&(x.close.shift(1)<=x.vwap.shift(1))&(x.close>x.vwap)&(x.body>=body)&(x.cloc>=.48)&(x.gap<=-trend)
          idx=np.flatnonzero(m.fillna(False).to_numpy());recent_hi=pd.concat([x.high.shift(k) for k in range(4)],axis=1).max(axis=1);recent_lo=pd.concat([x.low.shift(k) for k in range(4)],axis=1).min(axis=1);stop=(recent_hi.iloc[idx]+.04*x.atr.iloc[idx]).to_numpy() if side=='SHORT' else (recent_lo.iloc[idx]-.04*x.atr.iloc[idx]).to_numpy()
          specs.append((f'VWAP_{side}_b{body}_h{hour}_t{trend}_r{rr}',side,rr,idx,stop))
    # Trend pullback continuation. Entry only after completed rejection toward trend, never at the far edge.
    for side in ['SHORT','LONG']:
      for gap in [.20,.35,.55]:
       for pull in [.10,.25,.40]:
        for rr in [1.0,1.25,1.5,2.0]:
          if side=='LONG':m=base&(x.gap>=gap)&(x.e20d.between(-pull,.35))&(x.loc20.between(.30,.78))&(x.body>.02)&(x.cloc>=.56)&(x.close>x.ema9)
          else:m=base&(x.gap<=-gap)&(x.e20d.between(-.35,pull))&(x.loc20.between(.22,.70))&(x.body<-.02)&(x.cloc<=.44)&(x.close<x.ema9)
          idx=np.flatnonzero(m.fillna(False).to_numpy());recent_hi=pd.concat([x.high.shift(k) for k in range(3)],axis=1).max(axis=1);recent_lo=pd.concat([x.low.shift(k) for k in range(3)],axis=1).min(axis=1);stop=(recent_hi.iloc[idx]+.05*x.atr.iloc[idx]).to_numpy() if side=='SHORT' else (recent_lo.iloc[idx]-.05*x.atr.iloc[idx]).to_numpy()
          specs.append((f'PULLBACK_{side}_g{gap}_p{pull}_r{rr}',side,rr,idx,stop))
    return specs

def score_family(yearm):
    # must survive three distinct pre-holdout calendar years, not just pooled data
    if any(yearm.get(y,{}).get('n',0)<10 for y in [2023,2024,2025]):return -1e9
    if any(yearm[y]['expR']<=0 for y in [2023,2024,2025]):return -1e9
    if any(yearm[y]['PF']<1.05 for y in [2023,2024,2025]):return -1e9
    mn=min(yearm[y]['PF'] for y in [2023,2024,2025]); ex=min(yearm[y]['expR'] for y in [2023,2024,2025]); n=sum(yearm[y]['n'] for y in [2023,2024,2025])
    return .65*mn+1.7*ex+.00025*min(n,1000)

def schedule(events):
    if not events:return pd.DataFrame()
    q=pd.concat(events,ignore_index=True).sort_values(['entry_i','family_score'],ascending=[True,False])
    out=[];active=-1;dp=defaultdict(float);dn=defaultdict(int);dl=defaultdict(lambda:defaultdict(int));lastloss=defaultdict(set)
    for _,r in q.iterrows():
        ei=int(r.entry_i);date=r.date;side=r.side;fam=r.family
        if ei<=active:continue
        if dn[date]>=MAX_DAY or dp[date]<=DAYSTOP or dl[date][side]>=DIR_LOSS_KILL:continue
        if fam in lastloss[date]:continue
        out.append(r);nr=float(r.netR);dp[date]+=nr;dn[date]+=1
        if nr<0:dl[date][side]+=1;lastloss[date].add(fam)
        active=int(r.exit_i)
    return pd.DataFrame(out)

def portfolio_metric(t):
    if len(t)==0:return {'trades':0}
    s=t.netR.to_numpy(float);daily=t.groupby('date').netR.sum();gp=s[s>0].sum();gl=-s[s<=0].sum();res={'trades':len(t),'active_days':len(daily),'tpd':len(t)/len(daily),'netR':float(s.sum()),'PF':float(gp/gl) if gl else 99.0,'win_pct':float((s>0).mean()*100),'green_pct':float((daily>0).mean()*100),'avg_day_R':float(daily.mean()),'worst_day_R':float(daily.min()),'best_day_R':float(daily.max())}
    for side in ['LONG','SHORT']:
        z=t[t.side==side];res[side.lower()]={'trades':len(z),'netR':float(z.netR.sum()) if len(z) else 0,'PF':metric(z).get('PF',0) if len(z) else 0}
    res['dollars225']={'avg_day':res['avg_day_R']*RISK,'worst_day':res['worst_day_R']*RISK,'net':res['netR']*RISK}
    return res

def main():
    x=features(load()); specs=add_specs(x);rows=[];trades={}
    for name,side,rr,idx,stop in specs:
        t=outcome(x,idx,side,stop,rr);t['family']=name
        ym={y:metric(t[t.year==y]) for y in [2023,2024,2025]};sc=score_family(ym)
        rows.append({'family':name,'side':side,'rr':rr,'score':sc,'2023':ym[2023],'2024':ym[2024],'2025':ym[2025]})
        if sc>-1e8:trades[name]=t
    eligible=[r for r in rows if r['score']>-1e8]
    eligible.sort(key=lambda r:r['score'],reverse=True)
    # Greedy preholdout portfolio selection: up to 4 families/side, only if adding it improves minimum of 2024/2025 PF or avg-day expectancy.
    selected=[];best_quality=-1e9
    pool=eligible[:80]
    for side in ['LONG','SHORT']:
        chosen=[];sidepool=[r for r in pool if r['side']==side]
        for cand in sidepool:
            trial=chosen+[cand]
            ev=[]
            for z in trial:
                q=trades[z['family']].copy();q['family_score']=z['score'];ev.append(q[q.year<=2025])
            p=schedule(ev)
            m24=portfolio_metric(p[p.year==2024]);m25=portfolio_metric(p[p.year==2025]);
            if not m24.get('trades') or not m25.get('trades'):continue
            qual=min(m24.get('PF',0),m25.get('PF',0))+.55*min(m24.get('avg_day_R',-9),m25.get('avg_day_R',-9))-.03*abs(m25.get('tpd',0)-1.5)
            if not chosen or qual>best_quality+.025:
                chosen=trial;best_quality=qual
            if len(chosen)>=4:break
        selected.extend(chosen)
    # Fallback if one side has no eligible family: keep result explicitly unpromotable rather than force weak trades.
    evpre=[];evhold=[]
    for r in selected:
        q=trades[r['family']].copy();q['family_score']=r['score'];evpre.append(q[q.year<=2025]);evhold.append(q[q.year==2026])
    pre=schedule(evpre);hold=schedule(evhold)
    report={'bars':len(x),'start':str(x.index.min()),'end':str(x.index.max()),'specs_tested':len(rows),'eligible_stable_families':len(eligible),'selected_families':[{'family':r['family'],'side':r['side'],'score':r['score'],'2023':r['2023'],'2024':r['2024'],'2025':r['2025']} for r in selected],
      'preholdout_2023':portfolio_metric(pre[pre.year==2023]),'preholdout_2024':portfolio_metric(pre[pre.year==2024]),'validation_2025':portfolio_metric(pre[pre.year==2025]),'untouched_2026':portfolio_metric(hold),
      'promotion_pass':False,'warning':'2026 is used only after family selection. Research normalization $225/R is not live sizing.'}
    h=report['untouched_2026'];report['promotion_pass']=bool(h.get('PF',0)>=2 and h.get('avg_day_R',-9)>=1.78 and h.get('green_pct',0)>=60 and h.get('worst_day_R',-99)>=-3.15 and h.get('long',{}).get('netR',0)>0 and h.get('short',{}).get('netR',0)>0 and 2.0<=h.get('tpd',0)<=4.0)
    OUT.joinpath('V13_STRUCTURAL_REPORT.json').write_text(json.dumps(report,indent=2,default=str));pd.DataFrame([{**{k:v for k,v in r.items() if k not in ['2023','2024','2025']},'pf23':r['2023']['PF'],'exp23':r['2023']['expR'],'n23':r['2023']['n'],'pf24':r['2024']['PF'],'exp24':r['2024']['expR'],'n24':r['2024']['n'],'pf25':r['2025']['PF'],'exp25':r['2025']['expR'],'n25':r['2025']['n']} for r in rows]).sort_values('score',ascending=False).to_csv(OUT/'V13_FAMILY_GRID.csv',index=False)
    hold.to_csv(OUT/'V13_HOLDOUT_TRADES.csv',index=False);pre.to_csv(OUT/'V13_PREHOLDOUT_TRADES.csv',index=False)
    print(json.dumps(report,indent=2,default=str))
if __name__=='__main__':main()
