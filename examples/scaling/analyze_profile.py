#!/usr/bin/env python3
"""Summarise the in-pair phase profile (SO3_NATIVE_MPI_PROFILE) of the profiled
repeat: per-step totals, forward/backward/long-range phases, and the backward
pass split into compute and exchange. Usage: analyze_profile.py <model> [...]
(run from examples/scaling; reads results/<model>/ib/*/gpu_*/rep_1/slurm-*.out)."""
import re,glob,statistics as st,sys,collections
def kv(line): return {k:v for k,v in re.findall(r'(\w+)=([-\d.eE+]+)',line)}
def load(d):
    ph=collections.defaultdict(list); rv=collections.defaultdict(list)
    for f in glob.glob(d+'/slurm-*.out'):
        for l in open(f):
            if 'SO3LR_MPI_PHASE_PROFILE' in l:
                x=kv(l)
                if int(x['call'])>1: ph[int(x['call'])].append(x)
            elif 'SR_REVERSE_PROFILE' in l:
                x=kv(l)
                if int(x['call'])>1: rv[int(x['call'])].append(x)
    return ph,rv
def summarize(ph,rv):
    # per sampled call: mean over ranks; then mean over calls
    def m(dd,key,agg=st.mean):
        vals=[agg([float(r[key]) for r in rs]) for rs in dd.values() if all(key in r for r in rs)]
        return st.mean(vals) if vals else float('nan')
    out={k:m(ph,k) for k in ['total_ms','sr_forward_ms','sr_reverse_ms','native_lr_ms','upload_ms','download_ms','graph_plan_ms','topology_refresh_ms']}
    out['total_max']=m(ph,'total_ms',max)
    bl=[k for k in (next(iter(rv.values()))[0] if rv else {}) if re.match(r'block\d+_ms',k)]
    ex=[k for k in (next(iter(rv.values()))[0] if rv else {}) if re.match(r'exchange\d+_ms',k)]+['force_exchange_ms']
    out['rev_compute']=sum(m(rv,k) for k in bl)+m(rv,'seed_head_ms')+m(rv,'geometry_ms')
    out['rev_exchange']=sum(m(rv,k) for k in ex)
    return out
for model in sys.argv[1:]:
    for test in sorted(glob.glob(f'results/{model}/ib/*')):
        print(f"\n{model} {test.split('/')[-1]}:  GPUs | step total (mean/max rank) | fwd | rev | LR | rev compute | rev exchange (ms)")
        for g in sorted(glob.glob(test+'/gpu_*')):
            ph,rv=load(g+'/rep_1')
            if not ph: continue
            s=summarize(ph,rv)
            print(f"  {int(g[-3:]):3d} | {s['total_ms']:6.1f}/{s['total_max']:6.1f} | {s['sr_forward_ms']:6.1f} | {s['sr_reverse_ms']:6.1f} | {s['native_lr_ms']:5.1f} | {s['rev_compute']:6.1f} | {s['rev_exchange']:6.1f}")
