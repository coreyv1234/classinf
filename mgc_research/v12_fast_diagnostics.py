from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import v12_ml_router as v

OUT=Path('mgc_research/v12_diag'); OUT.mkdir(parents=True,exist_ok=True)

def stats(d):
    if len(d)==0:return {'n':0}
    s=d.netR.to_numpy(float); gp=s[s>0].sum(); gl=-s[s<=0].sum()
    return {'n':int(len(d)),'win_pct':float((s>0).mean()*100),'netR':float(s.sum()),'expR':float(s.mean()),'PF':float(gp/gl) if gl>0 else 99.0,'mean_prob':float(d.prob.mean())}

def main():
    x=v.feats(v.load()); base=v.candidates(x)
    out=[]
    for sm in v.STOP_MULTS:
      for rr in v.RRS:
        z=v.label(x,base,sm,rr); train=z[z.year<=2024]; val=z[z.year==2025]
        row={'stop_mult':sm,'rr':rr,'sides':{}}
        for side in ['LONG','SHORT']:
            model=v.fit_model(train,side); q=v.add_probs(model,val,side)
            auc=v.auc_score(q)
            qs={str(k):float(q.prob.quantile(k)) for k in [0.5,0.7,0.8,0.9,0.95,0.98,0.99]}
            bins=[]
            for pct in [0.50,0.30,0.20,0.10,0.05,0.02,0.01]:
                n=max(1,int(len(q)*pct)); g=q.nlargest(n,'prob'); bins.append({'top_fraction':pct,'threshold':float(g.prob.min()),**stats(g)})
            row['sides'][side]={'auc':auc,'prob_quantiles':qs,'top_probability_bins':bins}
        out.append(row)
    Path(OUT/'V12_FAST_DIAGNOSTICS.json').write_text(json.dumps({'bars':len(x),'base_candidates':len(base),'profiles':out},indent=2))
    print(json.dumps({'bars':len(x),'base_candidates':len(base),'profiles':out},indent=2))
if __name__=='__main__':main()
