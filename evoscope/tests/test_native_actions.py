from types import SimpleNamespace
import pytest
from evoscope.core import Artifacts, Bank, Policy, Task
from evoscope.environments import ToyEnvironment, WebShopEnvironment
from evoscope.engine import EvoScope, sample_checkpoints
from evoscope.runner import Runner, RestoreError
from evoscope.models import actor_call

class NativeFixture(ToyEnvironment):
    action_protocol='fixture-native-v1'
    def step(self,action):
        return self._view() if action=='fly' else super().step(action)


def test_agent_failure_retained_for_negative_probe(tmp_path):
    a=Artifacts(tmp_path/'a')
    model=SimpleNamespace(fingerprint='fixture',call=lambda system,data,phase,seed:
                          {'action':'fly' if data['policies'] else 'place'})
    task=Task('t','g','learn','place clean object',{'clean':True})
    bank=Bank([Policy('p','place clean object','always','fly')])
    runner=Runner(NativeFixture,model,a,max_steps=2)
    ep=runner.run(task,bank,'learn',0)
    assert ep.error is None and ep.score==0 and ep.actions==['fly','fly']
    checkpoints,coverage=sample_checkpoints([ep],'p',1)
    assert coverage['selected']['failure']==1
    engine=EvoScope(bank,runner,[task],a)
    evidence=engine.probe(checkpoints[0],'p',repeats=3)
    assert evidence['public']['direction']=='harmful'
    assert evidence['public']['mean_delta']==-1
    # Switching action semantics invalidates earlier checkpoints.
    runner.env_factory=ToyEnvironment
    with pytest.raises(RestoreError,match='protocol'):
        runner.run(task,bank,'probe',0,checkpoints[0],'p','mask')


@pytest.mark.parametrize('error',[ConnectionError,ValueError])
def test_infrastructure_or_protocol_failure_still_excluded(tmp_path,error):
    def fail(*args):raise error('private details')
    a=Artifacts(tmp_path/'a');model=SimpleNamespace(fingerprint='f',call=fail)
    bank=Bank([Policy('p','place','always','place')]);task=Task('t','g','learn','place')
    ep=Runner(NativeFixture,model,a,2).run(task,bank,'learn',0)
    assert ep.error==error.__name__
    assert sample_checkpoints([ep],'p',1)[0]==[]


def test_native_actions_still_require_json_schema():
    for value in ({'action':''},{'action':3},{'action':'look','extra':True}):
        with pytest.raises(ValueError):
            actor_call(SimpleNamespace(call=lambda *a:value),'',[],[],'learn',0,validate_actions=False)


def test_webshop_search_available_without_html_search_bar(tmp_path):
    e=WebShopEnvironment(str(tmp_path))
    e.env=SimpleNamespace(get_available_actions=lambda:{'has_search_bar':False,'clickables':['back to search']})
    assert 'search[<keywords>]' in e._view('results',0,False).admissible
