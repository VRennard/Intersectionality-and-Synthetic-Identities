"""Minor #5: is GPT-4o-mini's pair-cell missingness dimension-informative?
Failed (status='error') records retain demographics, so for every depth-2 cell
we tally error-rate by each involved dimension. If the rate is ~uniform across
dimensions, missingness is not dimension-biased and the collapse-direction
estimates (Fig 5) are unaffected."""
import json, os, glob
from collections import defaultdict
import numpy as np
EXCL={'49'}
dim_tot=defaultdict(int); dim_err=defaultdict(int)
pair_tot=0; pair_err=0
files=sorted(glob.glob('data/results/gpt-4o-mini/W*.jsonl'))
for f in files:
    w=os.path.basename(f)[1:-6]
    if w in EXCL: continue
    for ln in open(f):
        try: r=json.loads(ln)
        except: continue
        demo=r.get('demographics',[])
        if len(demo)!=2: continue
        # dimension = token before first space (e.g. 'Race Black' -> 'Race'); robust split
        dims=[d.rsplit(' ',1)[0] if ' ' in d else d for d in demo]
        bad = r.get('status')!='success'
        pair_tot+=1; pair_err+=bad
        for dm in dims:
            dim_tot[dm]+=1; dim_err[dm]+=bad
print(f"overall pair error-rate: {100*pair_err/pair_tot:.1f}% ({pair_err:,}/{pair_tot:,})\n")
print(f"{'dimension':22s} {'cells':>10s} {'err-rate':>9s}")
overall=pair_err/pair_tot
for dm in sorted(dim_tot, key=lambda d:-dim_err[d]/dim_tot[d]):
    rate=dim_err[dm]/dim_tot[dm]
    flag='  <-- ' + ('high' if rate>overall*1.25 else 'low' if rate<overall*0.8 else '') if abs(rate-overall)>0.03 else ''
    print(f"{dm:22s} {dim_tot[dm]:>10,} {100*rate:>8.1f}%{flag}")
rates=[dim_err[d]/dim_tot[d] for d in dim_tot]
print(f"\nspread across dimensions: {100*min(rates):.1f}% - {100*max(rates):.1f}%  (overall {100*overall:.1f}%)")
