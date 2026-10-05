"""Run with the deployed vLLM services; never deploy, train, or unseal Test implicitly."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import core  # Register the baseline task adapters without modifying baseline.
from . import common as C

def status(out):
    report = {'at':C.utc(),'tasks':{}}
    for task in C.TASKS:
        d = Path(out)/task
        report['tasks'][task] = {}
        for name in ('progress.json','statistics/train.json','statistics/dev.json','timing.json'):
            if (d/name).exists():
                report['tasks'][task][name] = C.read(d/name)
        report['tasks'][task]['test_state'] = 'sealed' if not (d/'evaluation_lock.json').exists() else 'evaluation_lock_present'
    return report

def models_with_identity(models):
    result = []
    for m in models:
        assets = {}
        p = Path(m['path'])
        for name in ('config.json','generation_config.json','tokenizer_config.json','model.safetensors.index.json'):
            if (p/name).exists():
                assets[name] = C.file_hash(p/name)
        result.append({**m,'identity':{'path':str(p),'config_hashes':assets}})
    return result

def run(out, pilot_only=False):
    from .collect import Collector, health, run_shard, verify_protocol, sealed_audit
    from .processing import clean, score, export
    import torch
    torch.set_num_threads(4)
    out = Path(out)
    if not (out/'prepared.json').exists():
        raise ValueError('Run prepare before collection')
    sizer = C.InputSizer()
    # Independent 128-source pilot for EVERY task before the full campaign.
    for task in C.TASKS:
        d = out/task
        p = verify_protocol(d,full=True)
        C.dump(d/'service_health.json',{'at':C.utc(),'checks':health(p)})
        if (d/'pilot_complete.json').exists():
            continue
        C.dump(d/'progress.json',{'stage':'pilot_generation','train_sources':128,'at':C.utc()})
        collectors = [Collector(d,m) for m in p['models']]
        run_shard(collectors,'train',0,128)
        clean(d,'train',sizer)
        score(d,'train')
        C.dump(d/'pilot_complete.json',C.read(d/'statistics/train.json'),immutable=True)
        C.summarize_timing(d)
        print(C.canonical({'task':task,'pilot':C.read(d/'pilot_complete.json')}),flush=True)
        del collectors
    if pilot_only:
        return
    for task in C.TASKS:
        d = out/task
        p = verify_protocol(d)
        if (d/'collection_complete.json').exists():
            continue
        collectors = [Collector(d,m) for m in p['models']]
        total, cursor, retained = p['pool_counts']['train'],128,0
        # First 5000 sources are mandatory; only then may retained-pair count stop expansion.
        while cursor < min(5000,total) or (retained < 100000 and cursor < total):
            stop = min(total,5000 if cursor < 5000 else total,cursor+500)
            C.dump(d/'progress.json',{'stage':'train_generation','source_start':cursor,'source_stop':stop,
                         'retained_before_shard':retained,'at':C.utc()})
            run_shard(collectors,'train',cursor,stop)
            summary = clean(d,'train',sizer)
            retained = summary['unique_valid_changed_pairs']
            C.summarize_timing(d)
            cursor = stop
        C.dump(d/'progress.json',{'stage':'train_scoring','train_sources':cursor,'at':C.utc()})
        score(d,'train')
        export(d,'train')
        for split in ('dev','test'):
            C.dump(d/'progress.json',{'stage':split+'_generation','at':C.utc()})
            run_shard(collectors,split,0,500)
            if split == 'test':
                sealed_audit(d)
            else:
                clean(d,split,sizer)
                score(d,split)
                export(d,split)
        report = {'task':task,'train_sources':cursor,'unique_valid_changed_pairs':retained,
                   'target_met':retained>=100000,'pool_exhausted':cursor==total,
                   'test':'sealed','finished_at_utc':C.utc()}
        C.dump(d/'collection_complete.json',report,immutable=True)
        C.dump(d/'progress.json',{'stage':'complete' if report['target_met'] else 'source_pool_exhausted',**report})
        C.summarize_timing(d)
        print(C.canonical(report),flush=True)
        del collectors
    C.dump(out/'campaign_complete.json',status(out),immutable=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','pilot','status','clean','score','export','sealed-audit'))
    parser.add_argument('--output',type=Path,default=C.OUT)
    parser.add_argument('--task',choices=C.TASKS)
    parser.add_argument('--split',choices=('train','dev','test'),default='train')
    args = parser.parse_args()
    if args.command == 'status':
        print(json.dumps(status(args.output),ensure_ascii=False,indent=2))
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command == 'prepare':
            from .prepare import prepare
            print(C.canonical(prepare(args.output)))
        elif args.command in ('run','pilot'):
            run(args.output,args.command=='pilot')
        else:
            if not args.task:
                parser.error('--task is required')
            d = args.output/args.task
            if args.command == 'sealed-audit':
                from .collect import sealed_audit
                print(C.canonical(sealed_audit(d)))
            else:
                from . import processing
                print(C.canonical(getattr(processing,args.command)(d,args.split)))

if __name__ == '__main__':
    main()
