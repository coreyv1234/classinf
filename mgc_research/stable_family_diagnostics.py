from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import stable_family_research as s

OUT=Path('mgc_research/stable_results'); OUT.mkdir(parents=True,exist_ok=True)

def flat(m,prefix):
    z={}
    for k,v in m.items():
        if isinstance(v,dict):
            for kk,vv in v.items(): z[f'{prefix}_{k}_{kk}']=vv
        else:z[f'{prefix}_{k}']=v
    return z

def main():
    x=s.feats(s.load()); e=s.events(x)
    rows=[]; famrows=[]; topfamilies=[]
    for name,sm,rr in s.PROFILES:
        t=s.label_profile(x,e,sm,rr); fs=s.family_stats(t)
        f=fs.copy(); f['profile']=name; f['stop_mult']=sm; f['rr']=rr
        f['min_pf_23_24']=f[['pf2023','pf2024']].min(axis=1)
        f['stable_score']=f['min_pf_23_24']+.35*f['expR']+.015*(f['n2023'].clip(upper=30)+f['n2024'].clip(upper=30))
        famrows.append(f)
        eligible=f[(f.n2023>=5)&(f.n2024>=5)].sort_values('stable_score',ascending=False).head(25)
        topfamilies.extend(eligible.to_dict('records'))
        for tier in ['STRICT','BALANCED']:
            sel=s.select_families(fs,tier)
            for slots in [1,2]:
                m23=s.metrics(s.simulate(t[t.year==2023],sel,slots)); m24=s.metrics(s.simulate(t[t.year==2024],sel,slots))
                if m23.get('trades',0) and m24.get('trades',0):
                    sc=min(m23.get('avg_day_R',-9),m24.get('avg_day_R',-9))+.25*min(m23.get('PF',0),m24.get('PF',0),3)+.003*min(m23.get('green_pct',0),m24.get('green_pct',0))
                else: sc=-999
                rows.append({'profile':name,'stop_mult':sm,'rr':rr,'tier':tier,'slots':slots,'families':len(sel),'diag_score':sc,**flat(m23,'y2023'),**flat(m24,'y2024')})
    grid=pd.DataFrame(rows).sort_values('diag_score',ascending=False)
    famall=pd.concat(famrows,ignore_index=True)
    stable=famall[(famall.n2023>=5)&(famall.n2024>=5)].sort_values('stable_score',ascending=False)
    grid.to_csv(OUT/'V11_PRE2025_DIAGNOSTIC_GRID.csv',index=False)
    famall.to_csv(OUT/'V11_PRE2025_FAMILY_STATS.csv',index=False)
    stable.head(250).to_csv(OUT/'V11_PRE2025_TOP_FAMILIES.csv',index=False)
    summary={'bars':len(x),'events':len(e),'top_families':stable.head(60).to_dict('records'),'note':'All ranking uses 2023-2024 only. 2025 and 2026 remain unread.'}
    OUT.joinpath('V11_PRE2025_DIAGNOSTIC.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__':main()
