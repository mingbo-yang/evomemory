"""Own two local model servers and four isolated workers for a bounded wall time."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import httpx
from .core import atomic_json

ROOT=Path(__file__).resolve().parents[1]
CONFIG={'glm':{'gpu':2,'port':8123,'kv':2147483648},'qwen':{'gpu':1,'port':8124,'kv':3221225472,'cpu_offload_gb':0},'qwen9b':{'gpu':0,'port':8125,'kv':3221225472,'cpu_offload_gb':0,'min_free_mib':26000,'gpu_memory_utilization':0.32}}

def stamp():return datetime.now(timezone.utc).isoformat()

def gpu_snapshot():
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
        proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
        return {'gpu':gpu,'processes':proc}
    except Exception as exc:return {'error':type(exc).__name__}

def stop(proc):
    # Session created by this watchdog; never target GPU-wide or name-wide processes.
    if getattr(proc, '_watchdog_cleaned', False):return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait(timeout=15)
    proc._watchdog_cleaned=True

def rows(path):
    if not path.exists():return []
    output=[]
    for line in path.read_text().splitlines():
        try:output.append(json.loads(line))
        except json.JSONDecodeError:pass  # The final line may be concurrently appended.
    return output

def progress(directory):
    result={}
    for dataset in ('alfworld','webshop'):
        base=directory/dataset;cost=[r for r in rows(base/'costs.jsonl') if r.get('kind')=='model']
        phases={}
        for path in base.glob('*/episodes.jsonl'):
            es=rows(path);valid=[e for e in es if not e.get('error')]
            phases[path.parent.name]={'completed':len(es),'errors':len(es)-len(valid),'successes':sum(e['score']>=1 for e in valid),'mean_score':sum(e['score'] for e in valid)/len(valid) if valid else None}
        result[dataset]={'model_calls':len(cost),'failed_calls':sum(not r.get('ok') for r in cost),'completion_tokens':sum(r.get('completion_tokens') or 0 for r in cost),'phases':phases,'full_summary_available':(directory/(dataset+'-summary.json')).exists()}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--hours',type=float,default=5);p.add_argument('--models',nargs='+',choices=('glm','qwen','qwen9b'));p.add_argument('--datasets',nargs='+',choices=('alfworld','webshop'),default=['alfworld','webshop']);args=p.parse_args()
    datasets=list(dict.fromkeys(args.datasets))
    configs={k:v for k,v in CONFIG.items() if k in (args.models or ['glm','qwen'])}
    if not 0<=args.hours<=5:raise ValueError('use 0 for no deadline, or 0 < hours <= 5')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(args.source).resolve();servers={};workers={};ready=set();failed=set();closed=set();started={};records=[]
    baseline=gpu_snapshot();start=time.monotonic();deadline=start+args.hours*3600 if args.hours else float('inf')
    atomic_json(out/'plan.json',{'started_utc':stamp(),'deadline_utc':(datetime.now(timezone.utc)+timedelta(hours=args.hours)).isoformat() if args.hours else None,'wall_hours':args.hours,'models':configs,'concurrent_datasets_per_model':len(datasets),'datasets':datasets,'source':str(source),'budget':'original predeclared task lists; stop early if all finish; preserve partial results at deadline','bank_isolation':'model x dataset','only_numeric_loopback':True,'baseline':baseline})
    atomic_json(out/'watchdog.json',{'pid':os.getpid(),'started_utc':stamp()})
    if (source/'plan.json').exists():
        protocol=json.loads((source/'plan.json').read_text());protocol['walltime_limit_hours']=args.hours
        protocol['concurrent_datasets_per_model']=len(datasets)
        # Deployment is specified per requested model; do not inherit GLM labels from the shared task plan.
        for key in ('model','gpu','cpu_offload_gb','kv_cache_bytes','gpu_memory_utilization'):
            protocol.pop(key,None)
        protocol['deployment_by_backend']=configs
        atomic_json(out/'experiment-protocol.json',protocol)
    provenance=ROOT/'downloads/WebShop-10k/data/provenance.json'
    if provenance.exists():shutil.copy2(provenance,out/'webshop-data-provenance.json')
    snapshots=out/'source';snapshots.mkdir()
    for filename in ('local_watchdog.py','local_expanded.py','environments.py','engine.py','gate_plan.py','runner.py','core.py','models.py','editor_budget.py'):
        shutil.copy2(ROOT/'evoscope'/filename,snapshots/filename)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def log(event,**data):
        row={'time':stamp(),'event':event,**data}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        with httpx.Client(trust_env=False,timeout=2) as client:
            # Preflight every port before owning any model server.
            for backend,cfg in configs.items():
                try:client.get(f"http://127.0.0.1:{cfg['port']}/health")
                except httpx.ConnectError:pass
                else:raise RuntimeError(f'{backend} port already occupied')
            for backend,cfg in configs.items():
                folder=out/backend;folder.mkdir()
                for filename in [f'{d}-{suffix}.json' for d in datasets for suffix in ('manifest','m0')]:
                    shutil.copy2(source/filename,folder/filename)
                script=ROOT/f'evoscope/scripts/serve_{backend}_local.sh';shutil.copy2(script,snapshots/script.name)
                env=os.environ.copy()
                for key in ('EVOSCOPE_API_KEY','EVOSCOPE_CREDENTIALS_FILE','OPENAI_API_KEY','OPENAI_BASE_URL'):env.pop(key,None)
                env.update(EVOSCOPE_GPU_MEMORY_UTILIZATION=str(cfg.get('gpu_memory_utilization',0.66)),EVOSCOPE_CPU_OFFLOAD_GB=str(cfg.get('cpu_offload_gb',8)),EVOSCOPE_KV_CACHE_BYTES=str(cfg['kv']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR='/tmp')
                free=int(subprocess.check_output(['nvidia-smi',f"--id={cfg['gpu']}",'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<cfg.get('min_free_mib',57000):
                    failed.add(backend);log('resource_preflight_failed',backend=backend,free_mib=free);continue
                with (folder/'server.log').open('w') as f:
                    proc=subprocess.Popen(['bash',str(script)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                servers[backend]=proc;started[backend]=time.monotonic()
                records.append({'role':'server','backend':backend,'pid':proc.pid,'pgid':proc.pid,'gpu':cfg['gpu']});atomic_json(out/'owned-processes.json',records)
                log('server_start',backend=backend,pid=proc.pid)
            last_status=0
            while time.monotonic()<deadline:
                for worker in workers.values():
                    if worker.poll() is not None:stop(worker)
                for backend,proc in servers.items():
                    if backend in closed:continue
                    if proc.poll() is not None:
                        failed.add(backend);log('server_exit',backend=backend,returncode=proc.returncode)
                        for (b,d),worker in workers.items():
                            if b==backend and worker.poll() is None:stop(worker)
                        stop(proc);closed.add(backend);continue
                    if backend not in ready:
                        try:healthy=client.get(f"http://127.0.0.1:{configs[backend]['port']}/health").status_code==200
                        except httpx.HTTPError:healthy=False
                        if healthy:
                            ready.add(backend);log('server_ready',backend=backend)
                            for dataset in datasets:
                                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',TMPDIR='/tmp',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                                for key in ('EVOSCOPE_API_KEY','EVOSCOPE_CREDENTIALS_FILE','OPENAI_API_KEY','OPENAI_BASE_URL'):env.pop(key,None)
                                cmd=[sys.executable,'-u','-m','evoscope.local_expanded','--output',str(out/backend),'--backend',backend,'--worker-dataset',dataset]
                                with (out/backend/(dataset+'-worker.log')).open('w') as f:
                                    worker=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                                workers[backend,dataset]=worker;records.append({'role':'worker','backend':backend,'dataset':dataset,'pid':worker.pid,'pgid':worker.pid});atomic_json(out/'owned-processes.json',records)
                                log('worker_start',backend=backend,dataset=dataset,pid=worker.pid)
                        elif time.monotonic()-started[backend]>1800:
                            failed.add(backend);stop(proc);closed.add(backend);log('startup_timeout',backend=backend)
                    elif all(workers[backend,d].poll() is not None for d in datasets):
                        stop(proc);closed.add(backend);log('model_finished_and_released',backend=backend)
                if time.monotonic()-last_status>=60:
                    state={'updated_utc':stamp(),'elapsed_seconds':time.monotonic()-start,'remaining_seconds':max(0,deadline-time.monotonic()) if args.hours else None,'gpu':gpu_snapshot(),'progress':{b:progress(out/b) for b in configs},'workers':{f'{b}/{d}':w.poll() for (b,d),w in workers.items()},'failed_models':sorted(failed)}
                    atomic_json(out/'status.json',state);last_status=time.monotonic()
                    log('progress',calls={b:sum(v['model_calls'] for v in state['progress'][b].values()) for b in configs})
                if all(b in closed or b in failed for b in configs):break
                time.sleep(5)
            reason='walltime_limit' if time.monotonic()>=deadline else 'all_processes_finished'
            atomic_json(out/'stop-reason.json',{'reason':reason,'time':stamp()})
    except BaseException as exc:
        atomic_json(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'time':stamp()});raise
    finally:
        for worker in workers.values():stop(worker)
        # Capture GPU-owning descendants before shutting down the known process groups.
        owned_gpu=[]
        for line in gpu_snapshot().get('processes','').splitlines():
            try:
                pid=int(line.split(',')[0]);pgid=os.getpgid(pid)
                if pgid in {p.pid for p in servers.values()}:owned_gpu.append(pid)
            except (ValueError,ProcessLookupError):pass
        for proc in servers.values():stop(proc)
        time.sleep(3);after=gpu_snapshot()
        remaining={int(line.split(',')[0]) for line in after.get('processes','').splitlines() if line.strip()}
        atomic_json(out/'cleanup.json',{'time':stamp(),'owned_gpu_pids':owned_gpu,'owned_gpu_pids_remaining':sorted(set(owned_gpu)&remaining),'verification_available':'error' not in after,'before':baseline,'after':after})
        final={'time':stamp(),'elapsed_seconds':time.monotonic()-start,'progress':{b:progress(out/b) for b in configs},'workers':{f'{b}/{d}':w.poll() for (b,d),w in workers.items()},'all_requested_completed':len(workers)==len(datasets)*len(configs) and all(w.returncode==0 for w in workers.values()),'failed_models':sorted(failed),'gpu_cleanup_verified':'error' not in after and not set(owned_gpu)&remaining}
        atomic_json(out/'final-status.json',final)
        report=['# 五小时本地实验记录','',f'完整完成请求的模型/数据集组合：{final["all_requested_completed"]}。本进程 GPU 显存释放验证：{final["gpu_cleanup_verified"]}。','', '未完成的实验只报告已完成轨迹，不推断方法有效性。','', '| 模型 | 数据集 | 完成模型调用 | 失败调用 | 完整比较 |','|---|---|---:|---:|---|']
        for b,ds in final['progress'].items():
            for d,v in ds.items():report.append(f'| {b} | {d} | {v["model_calls"]} | {v["failed_calls"]} | {v["full_summary_available"]} |')
        (out/'REPORT.md').write_text('\n'.join(report)+'\n');log('cleanup_complete',verified=final['gpu_cleanup_verified'])

if __name__=='__main__':main()
