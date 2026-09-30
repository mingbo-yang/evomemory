import copy
import json
from types import SimpleNamespace as S
from evoscope.editor_budget import prepare,MAX_BYTES
from evoscope.engine import spread_probe_schedule
from evoscope.core import Bank,Policy,atomic_json


def test_compression_preserves_every_signed_pair_and_is_deterministic():
    state={'observation':'large observation '*10000,'action':'look','admissible':['look']}
    evidence=[{'goal':'find object','history':[state]*30,'pairs':[
        {'expose':{'score':x,'done':False,'continuation':[state]*30},'mask':{'score':y,'done':True,'continuation':[state]*25}}
        for x,y in [(0,1),(1,0),(.5,.5)]],'direction':'unstable','mean_delta':0}]
    original=copy.deepcopy(evidence);policy={'key':'look','when':'visible','do':'look'}
    payload,audit=prepare(policy,evidence)
    assert evidence==original and payload['policy']==policy
    assert len(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode())<=MAX_BYTES
    assert [(p['expose']['score'],p['mask']['score']) for p in payload['evidence'][0]['pairs']]==[(0,1),(1,0),(.5,.5)]
    assert prepare(policy,evidence)==(payload,audit)
    assert len(payload['traces'])<7


def test_scheduled_budget_covers_early_middle_late_and_policy_caps():
    schedule=spread_probe_schedule(25,2,2,(6,3))
    assert sum(schedule.values())==6
    assert sum(v for r,v in schedule.items() if r%2==0)==3
    assert sum(v for r,v in schedule.items() if r%2==1)==3
    assert any(r<8 for r in schedule) and any(8<=r<17 for r in schedule) and any(r>=17 for r in schedule)
    assert max(schedule.values())<=2
    assert spread_probe_schedule(3,2,2,(0,3))=={}


def test_episode_failure_does_not_abort_other_arms_or_tasks(tmp_path,monkeypatch):
    import evoscope.local_expanded as mod
    bank=Bank([Policy('p','look','visible','look')]);atomic_json(tmp_path/'alfworld-m0.json',bank.to_json())
    tasks=[{'id':f't{i}','group_id':f'g{i}','split':'test','goal':'look','payload':{},'seed':42} for i in range(2)]
    atomic_json(tmp_path/'alfworld-manifest.json',tasks);calls=[]
    monkeypatch.setattr(mod,'LocalModel',lambda *a,**k:S(config={}))
    class FakeRunner:
        def __init__(self,*a):pass
        def run(self,t,b,phase,seed):
            calls.append((t.id,phase))
            return S(task_id=t.id,score=0 if len(calls)==1 else 1,error='JSONDecodeError' if len(calls)==1 else None,bank_hash=b.fingerprint)
    monkeypatch.setattr(mod,'Runner',FakeRunner)
    class FakeEngine:
        def __init__(self,b,*args):self.bank=b;self.probe_attempts=[]
        def evolve(self,**kwargs):return {'first_run':{'errors':1},'accepted_edits':0}
    monkeypatch.setattr(mod,'EvoScope',FakeEngine)
    result=mod.experiment(tmp_path,'alfworld','fake','http://127.0.0.1:8123/v1')
    assert len(calls)==6
    assert result['test']['no-memory']['episodes']==2
    assert result['test']['no-memory']['errors']==1
    assert result['test']['no-memory']['success_rate_with_errors_as_failures']==.5


def test_editor_exact_token_preflight_rejects_without_generation(tmp_path):
    import pytest
    from evoscope.core import Artifacts
    from evoscope.models import edit_condition
    model=S(artifacts=Artifacts(tmp_path/'budget'),max_tokens=8192,
            count_editor_tokens=lambda *args:(25000,32768),
            call=lambda *args:pytest.fail('over-budget request was sent'))
    with pytest.raises(ValueError,match='context budget'):
        edit_condition(model,{'key':'k','when':'w','do':'d'},[],0)


def test_invalid_json_is_saved_and_counted_as_failure(tmp_path):
    import pytest
    from evoscope.core import Artifacts
    from evoscope.models import APIModel
    model=object.__new__(APIModel);model.artifacts=Artifacts(tmp_path/'raw')
    model.model='fake';model.config={};model.send_seed=True;model.temperature=0;model.max_tokens=8192
    response=S(choices=[S(message=S(content='not json'),finish_reason='stop')],usage=None,model='fake',system_fingerprint=None)
    model.client=S(chat=S(completions=S(create=lambda **kwargs:response)))
    with pytest.raises(json.JSONDecodeError):model.call('system',{},'test',3)
    row=json.loads((model.artifacts.root/'test/invalid_outputs.jsonl').read_text())
    assert row['content']=='not json'
    assert model.artifacts.costs[-1]['ok'] is False
