"""Predeclared local benchmark expansion, dataset-isolated banks and owned-server cleanup."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import zipfile
from .core import Artifacts, Bank, atomic_json, load_manifest, validate_tasks
from .engine import EvoScope, summarize
from .environments import AlfWorldEnvironment, WebShopEnvironment
from .models import LocalModel
from .runner import Runner

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'evoscope/results/local-glm-20260924'

def gpu_processes():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)

def prepare(out):
    rng=random.Random(20260925)
    old=load_manifest(PRIOR/'alfworld-manifest.json')
    excluded={t.group_id for t in old}; tasks=[]
    games=out/'alfworld-games';games.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'downloads/alfworld-tw-0.4.2.zip') as archive:
        names=sorted(n for n in archive.namelist() if n.endswith('/game.tw-pddl'))
        families=sorted({n.split('/')[-3].split('-')[0] for n in names})
        for split,source,per_family in [('learn','train',2),('gate','train',1),('test','valid_unseen',2)]:
            for family in families:
                candidates={}
                for name in names:
                    group=name.split('/')[-3]
                    if name.split('/')[1]==source and group.startswith(family+'-') and group not in excluded:
                        candidates.setdefault(group,name)
                groups=sorted(candidates);rng.shuffle(groups)
                assert len(groups)>=per_family
                for group in groups[:per_family]:
                    excluded.add(group);idx=len(tasks);path=games/f'{split}-{idx}.tw-pddl';path.write_bytes(archive.read(candidates[group]))
                    tasks.append(dict(id=f'{split}-{idx}',group_id=group,split=split,goal='Complete the household task stated in the initial observation.',payload={'gamefile':str(path.resolve())},seed=42))
    atomic_json(out/'alfworld-manifest.json',tasks)
    old=load_manifest(PRIOR/'webshop-manifest.json');excluded={t.group_id for t in old}
    goals=json.loads((out/'webshop-goal-index.json').read_text());by_asin={}
    for g in goals:
        if 'product-'+g['asin'] not in excluded:by_asin.setdefault(g['asin'],g['index'])
    asins=sorted(by_asin);rng.shuffle(asins);assert len(asins)>=30
    tasks=[];offset=0
    for split,count in [('learn',12),('gate',6),('test',12)]:
        for asin in asins[offset:offset+count]:
            tasks.append(dict(id=f'{split}-{len(tasks)}',group_id='product-'+asin,split=split,goal='Complete the shopping instruction in the initial observation.',payload={'goal_index':by_asin[asin]},seed=42))
        offset+=count
    atomic_json(out/'webshop-manifest.json',tasks)
    for dataset,source in [('alfworld',ROOT/'evoscope/results/local-glm-fixes-32k-20260924/alfworld_m0.json'),('webshop',PRIOR/'webshop-pilot/m0/bank.json')]:
        bank=Bank.load(source);atomic_json(out/f'{dataset}-m0.json',bank.to_json())
        validate_tasks(load_manifest(out/f'{dataset}-manifest.json'))
    atomic_json(out/'plan.json',{'local_only':True,'gpu':2,'model':'GLM-4.7-Flash','seed':20260925,
        'datasets':['alfworld','webshop'],'per_dataset':{'learn':12,'gate':6,'test':12},
        'alfworld_stratification':'six families, train learn/gate and valid_unseen test; previous groups excluded',
        'webshop':'10000-product community mirror, ASIN-disjoint splits; previous ASINs excluded',
        'banks':'separate per dataset, immutable pre-generated natural M0; updates carry only within dataset',
        'probes_per_dataset':6,'probes_per_policy':3,'probe_repeats':3,'gate_repeats':3,'gate_groups_per_edit':2,
        'evaluation_repeats':1,'arms':['no-memory','frozen-M0','EvoScope'],'max_tokens':8192,
        'max_steps':{'alfworld':30,'webshop':20},'gpu_memory_utilization':0.66,'cpu_offload_gb':12,
        'limitation':'Expanded exploratory experiment, not a multi-seed significance study; no test-driven edits.'})

def stratified_test_summary(episodes, tasks):
    """Keep generalization strata separate and weight independent groups equally."""
    from collections import defaultdict
    from statistics import mean
    lookup={t.id:t for t in tasks};result={}
    for arm,items in episodes.items():
        strata=defaultdict(list)
        for ep in items:strata[lookup[ep.task_id].payload.get('evaluation_partition','default')].append(ep)
        result[arm]={}
        for partition,es in strata.items():
            groups=defaultdict(list)
            for ep in es:groups[lookup[ep.task_id].group_id].append(ep)
            result[arm][partition]={**summarize(es),'independent_groups':len(groups),
                'group_macro_success_rate_errors_as_failures':mean(mean(float(e.score>=1 and not e.error) for e in group) for group in groups.values()),
                'note':'Trials sharing a group are not independent; report this partition separately.'}
    return result


def experiment(out,dataset,model_name,base_url):
    artifacts=Artifacts(out/dataset)
    tasks=load_manifest(out/f'{dataset}-manifest.json')
    bank_path=out/f'{dataset}-m0.json';bank=Bank.load(bank_path);hash0=bank.fingerprint
    artifacts.write('initial_bank_source',{'path':str(bank_path),'hash':hash0})
    max_steps=30 if dataset=='alfworld' else 20
    factory=(lambda:AlfWorldEnvironment(30)) if dataset=='alfworld' else (lambda:WebShopEnvironment(str(ROOT/'downloads/WebShop-10k'),num_products=None))
    model=LocalModel(artifacts,model_name,base_url,max_tokens=8192,send_seed=True)
    artifacts.write('protocol',{'version':'evoscope-protocol-v3','environment_resources':'alfworld-planner-close-v1','editor':'editor-budget-v1','episode_errors':'continue without retry; report as failures','probe_schedule':'evenly-spaced-rounds-v1','gate':'candidate-independent-local-global-v1','sampling':'cross-round-policy-version-v1','evidence':'multi-state-bank-context-v1','min_edit_checkpoints':3,'min_non_neutral':1,'probe_limits':[12,6],'gate_size':4,'gate_local_size':2,'max_candidates_per_policy':2})
    artifacts.write('model_config',model.config)
    runner=Runner(factory,model,artifacts,max_steps)
    engine=EvoScope(bank,runner,tasks,artifacts);engine.probe_limits=(12,6)
    print('START EVOLUTION',dataset,flush=True)
    evolution=engine.evolve(batch_size=4,probe_checkpoints=2,repeats=3,gate_size=4,seed=42,gate_repeats=3,gate_local_size=2,max_candidates_per_policy=2)
    print('EVOLUTION DONE',dataset,json.dumps(evolution),flush=True)
    artifacts.write('evolution_summary',evolution)
    banks={'no-memory':Bank([]),'frozen-M0':Bank.load(bank_path),'EvoScope':engine.bank}
    episodes={k:[] for k in banks};pairs=[]
    for i,task in enumerate(t for t in tasks if t.split=='test'):
        row={'task_id':task.id,'group_id':task.group_id,'seed':500000+i*100}
        order=list(banks);order=order[i%3:]+order[:i%3]
        row['order']=order
        for arm in order:
            ep=runner.run(task,banks[arm],'test-'+arm,row['seed']);episodes[arm].append(ep)
            row[arm]={'score':ep.score,'error':ep.error,'bank_hash':ep.bank_hash}
            print('TEST',dataset,task.id,arm,row[arm],flush=True)
            if ep.error:artifacts.append('comparison/execution_errors',{'task_id':task.id,'arm':arm,'error':ep.error,'seed':row['seed'],'action':'record failure and continue; no retry'})
        pairs.append(row);artifacts.append('comparison/pairs',row)
        artifacts.write('comparison/progress',{arm:summarize(eps) for arm,eps in episodes.items()})
    artifacts.write('comparison/stratified',stratified_test_summary(episodes,tasks))
    assert Bank.load(bank_path).fingerprint==hash0
    result={'dataset':dataset,'initial_bank_hash':hash0,'evolution':evolution,
        'probes':engine.probe_attempts,'test':{arm:summarize(eps) for arm,eps in episodes.items()},
        'costs':artifacts.cost_summary(),'test_used_for_updates':False}
    artifacts.write('comparison/summary',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--backend',choices=('glm','qwen','qwen9b'),default='glm');p.add_argument('--worker-dataset',choices=('alfworld','webshop'));args=p.parse_args()
    out=Path(args.output).resolve()
    if args.prepare_only:prepare(out);return
    if args.worker_dataset:
        model_name={'glm':'glm-4.7-flash-local','qwen':'qwen3.8-27b-local','qwen9b':'qwen3.5-9b-local'}[args.backend]
        port={'glm':8123,'qwen':8124,'qwen9b':8125}[args.backend]
        result=experiment(out,args.worker_dataset,model_name,f'http://127.0.0.1:{port}/v1')
        atomic_json(out/(args.worker_dataset+'-summary.json'),result)
        return
    if args.backend=='qwen9b':raise ValueError('qwen9b requires local_watchdog --models qwen9b')
    if (out/'server.json').exists():raise ValueError('use a fresh run; refuse overwrite')
    gpu=2 if args.backend=='glm' else 1
    port={'glm':8123,'qwen':8124,'qwen9b':8125}[args.backend]
    model_name={'glm':'glm-4.7-flash-local','qwen':'qwen3.8-27b-local','qwen9b':'qwen3.5-9b-local'}[args.backend]
    base_url=f'http://127.0.0.1:{port}/v1'
    script=ROOT/f'evoscope/scripts/serve_{args.backend}_local.sh'
    server=None;results={};baseline=gpu_processes()
    def interrupt(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGALRM,interrupt);signal.alarm(86400)
    try:
        import httpx
        with httpx.Client(trust_env=False,timeout=2) as client:
            try:client.get(f'http://127.0.0.1:{port}/health')
            except httpx.ConnectError:pass
            else:raise RuntimeError('port occupied; will not manage existing server')
            free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<57000:raise RuntimeError(f'not enough free VRAM for shared configuration: {free} MiB')
            env=os.environ.copy();env['EVOSCOPE_GPU_MEMORY_UTILIZATION']='0.66';env['EVOSCOPE_CPU_OFFLOAD_GB']='8';env['EVOSCOPE_KV_CACHE_BYTES']='2147483648' if args.backend=='glm' else '3221225472'
            for key in ('EVOSCOPE_API_KEY','EVOSCOPE_CREDENTIALS_FILE','OPENAI_API_KEY','OPENAI_BASE_URL'):env.pop(key,None)
            with (out/'server.log').open('w') as log:
                server=subprocess.Popen(['bash',str(script)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(out/'server.json',{'pid':server.pid,'baseline_processes':baseline,'startup_script':script.read_text(),'gpu':gpu,'backend':args.backend,'memory_utilization':0.66,'cpu_offload_gb':8,'kv_cache_bytes':env['EVOSCOPE_KV_CACHE_BYTES']})
            startup=time.monotonic()+1800
            while time.monotonic()<startup:
                if server.poll() is not None:raise RuntimeError('server startup failed')
                try:
                    if client.get(f'http://127.0.0.1:{port}/health').status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(2)
            else:raise RuntimeError('server startup timeout')
        print('SERVER READY',flush=True)
        for dataset in ('alfworld','webshop'):
            results[dataset]=experiment(out,dataset,model_name,base_url);atomic_json(out/'summary.json',results)
    except BaseException as exc:
        atomic_json(out/'failure.json',{'type':type(exc).__name__,'message':str(exc)});raise
    finally:
        signal.alarm(0)
        if server is not None:
            try:os.killpg(server.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:pass
            try:os.killpg(server.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            server.wait();time.sleep(3)
        atomic_json(out/'cleanup.json',{'owned_server_pid':server.pid if server else None,'returncode':server.returncode if server else None,'processes_after':gpu_processes(),'baseline_processes':baseline})
        print('CLEANUP COMPLETE',flush=True)

if __name__=='__main__':main()
