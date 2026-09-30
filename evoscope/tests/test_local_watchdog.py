import itertools
import json
from pathlib import Path
import signal
from types import SimpleNamespace
from unittest.mock import Mock

from evoscope import local_watchdog as watchdog


def test_9b_uses_card_zero_separate_port_without_cpu_offload():
    cfg = watchdog.CONFIG['qwen9b']
    assert cfg['gpu'] == 0 and cfg['cpu_offload_gb'] == 0
    assert cfg['port'] not in {watchdog.CONFIG['qwen']['port'], watchdog.CONFIG['glm']['port']}


def test_stop_targets_only_owned_group_and_is_idempotent(monkeypatch):
    signals=[]
    monkeypatch.setattr(watchdog.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    proc=SimpleNamespace(pid=12345,wait=lambda timeout:0)
    watchdog.stop(proc)
    watchdog.stop(proc)
    assert signals==[(12345,signal.SIGTERM),(12345,signal.SIGKILL)]


def test_progress_counts_completed_episodes_not_partial_actions(tmp_path):
    base=tmp_path/'alfworld';(base/'learn').mkdir(parents=True)
    (base/'costs.jsonl').write_text(json.dumps({'kind':'model','ok':True,'completion_tokens':7})+'\n'+ '{"kind":')
    (base/'learn/episodes.jsonl').write_text(json.dumps({'score':1,'error':None})+'\n'+json.dumps({'score':0,'error':'APIError'})+'\n')
    data=watchdog.progress(tmp_path)['alfworld']
    assert data['model_calls']==1
    assert data['phases']['learn']=={'completed':2,'errors':1,'successes':1,'mean_score':1}
    assert not data['full_summary_available']


def test_deadline_stops_owned_server_without_starting_workers(tmp_path,monkeypatch):
    source=tmp_path/'inputs';source.mkdir()
    for name in ('alfworld-manifest.json','webshop-manifest.json','alfworld-m0.json','webshop-m0.json','split-audit.json'):
        (source/name).write_text('[]')
    monkeypatch.setattr(watchdog,'CONFIG',{'glm':{'gpu':2,'port':8123,'kv':2147483648}})
    monkeypatch.setattr(watchdog,'gpu_snapshot',lambda:{'gpu':'','processes':''})
    monkeypatch.setattr(watchdog.subprocess,'check_output',lambda *a,**k:'70000')
    signals=[]
    monkeypatch.setattr(watchdog.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    monkeypatch.setattr(watchdog.signal,'signal',lambda *a:None)
    proc=SimpleNamespace(pid=4321,wait=lambda timeout:0,returncode=0,poll=lambda:None)
    popen=Mock(return_value=proc);monkeypatch.setattr(watchdog.subprocess,'Popen',popen)
    class Client:
        def __init__(self,**kwargs):assert kwargs['trust_env'] is False
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,url):raise watchdog.httpx.ConnectError('absent')
    monkeypatch.setattr(watchdog.httpx,'Client',Client)
    clock=itertools.count();monkeypatch.setattr(watchdog.time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(watchdog.time,'sleep',lambda *args:None)
    out=tmp_path/'run'
    monkeypatch.setattr(watchdog.sys,'argv',['watchdog','--output',str(out),'--source',str(source),'--hours','0.0001'])
    watchdog.main()
    assert popen.call_count==1
    assert signals==[(4321,signal.SIGTERM),(4321,signal.SIGKILL)]
    assert json.loads((out/'stop-reason.json').read_text())['reason']=='walltime_limit'
    assert not json.loads((out/'final-status.json').read_text())['all_requested_completed']
