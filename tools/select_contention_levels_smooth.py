#!/usr/bin/env python3
import argparse,itertools,json,math,re
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--candidate-dir',required=True); p.add_argument('--window-ms',type=float,required=True); p.add_argument('--slice-ms',type=float,required=True); p.add_argument('--start-delay-ms',type=float,required=True); p.add_argument('--threads',type=int,required=True); p.add_argument('--targets',default='1.20,1.45,1.80,2.30'); p.add_argument('--min-p50-gap-ms',type=float,default=0.05); p.add_argument('--output',required=True); a=p.parse_args()
    b=json.load(open(a.baseline)); base=float(b['probe']['p50_ms']); rows=[]; pat=re.compile(r'probe_s([0-9p]+)\.json$')
    for f in sorted(Path(a.candidate_dir).glob('probe_s*.json')):
        m=pat.search(f.name)
        if not m: continue
        d=json.load(open(f)); s=float(m.group(1).replace('p','.')); rows.append({'strength':s,'probe_p10_ms':d['probe']['p10_ms'],'probe_p50_ms':d['probe']['p50_ms'],'probe_p90_ms':d['probe']['p90_ms'],'probe_p99_ms':d['probe']['p99_ms'],'probe_std_ms':d['probe']['std_ms'],'slowdown':d['probe']['p50_ms']/base,'chain_p50_ms':None if d['contention_chain_actual'] is None else d['contention_chain_actual']['p50_ms'],'raw_ms':d['raw_ms'],'contention':d['contention']})
    rows.sort(key=lambda r:r['strength']); env=[]; best=base
    for r in rows:
        if r['probe_p50_ms']>=best+a.min_p50_gap_ms: env.append(r); best=r['probe_p50_ms']
    if len(env)<4: raise RuntimeError('Fewer than 4 separated points: '+str([(r['strength'],round(r['probe_p50_ms'],4),round(r['slowdown'],3)) for r in rows]))
    targets=[float(x) for x in a.targets.split(',')]; bestc=None; bestcost=1e99
    for combo in itertools.combinations(env,4):
        cost=sum((math.log(max(c['slowdown'],1e-6))-math.log(t))**2 for c,t in zip(combo,targets))
        if cost<bestcost: bestcost=cost; bestc=combo
    levels=[{'level':'L0','strength':0.0,'probe_p10_ms':b['probe']['p10_ms'],'probe_p50_ms':b['probe']['p50_ms'],'probe_p90_ms':b['probe']['p90_ms'],'probe_p99_ms':b['probe']['p99_ms'],'probe_std_ms':b['probe']['std_ms'],'slowdown':1.0,'raw_ms':b['raw_ms'],'contention':None}]
    for i,r in enumerate(bestc,1): levels.append({'level':f'L{i}',**r})
    cents=[x['probe_p50_ms'] for x in levels]; conf=[[0]*5 for _ in range(5)]; total=correct=0; acc={}
    for ti,l in enumerate(levels):
        for x in l['raw_ms']:
            pi=min(range(5),key=lambda j:abs(x-cents[j])); conf[ti][pi]+=1; total+=1; correct+=pi==ti
        acc[l['level']]=conf[ti][ti]/len(l['raw_ms'])
    compact=[]
    for l in levels: q=dict(l); q.pop('raw_ms',None); compact.append(q)
    res={'workload':{'type':'short_kernel_microchain_compute_contention','vary_only':'strength=blocks/SM','window_ms':a.window_ms,'slice_ms':a.slice_ms,'start_delay_ms':a.start_delay_ms,'threads':a.threads},'targets':targets,'levels':compact,'nearest_centroid_thresholds_ms':[(cents[i]+cents[i+1])/2 for i in range(4)],'classifier_validation':{'centroids_ms':cents,'confusion_matrix_rows_true_cols_pred':conf,'per_level_accuracy':acc,'overall_accuracy':correct/total}}
    Path(a.output).write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
