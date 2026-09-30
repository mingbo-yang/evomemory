import json
import sys
from types import SimpleNamespace
import pytest
from evoscope.core import Artifacts, Task, validate_tasks
from evoscope.environments import WebShopEnvironment
from evoscope.models import LocalModel, actor_call


def test_local_never_reads_gateway_credentials(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv('EVOSCOPE_API_KEY', 'remote-secret-must-not-be-used')
    monkeypatch.setenv('EVOSCOPE_CREDENTIALS_FILE', '/does/not/exist')
    monkeypatch.setenv('HTTP_PROXY', 'http://unrelated.invalid:1234')
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=lambda **kw: captured.update(kw)))
    model = LocalModel(Artifacts(tmp_path/'local'), 'local-glm', 'http://127.0.0.1:8123/v1')
    assert captured['api_key'] == 'local-not-a-secret'
    assert not captured['http_client'].follow_redirects
    assert not captured['http_client']._trust_env
    assert 'remote-secret' not in json.dumps(model.config)
    captured['http_client'].close()
    for url in ('http://172.25.76.237:3000/v1', 'https://api.openai.com/v1',
                'http://127.0.0.1:8123/v1?redirect=elsewhere', 'http://secret@127.0.0.1/v1'):
        with pytest.raises(ValueError, match='loopback'): LocalModel(None, 'glm', url)


@pytest.mark.parametrize('action,allowed', [('search[red running shoes]', True),
    ('click[buy now]', True), ('click[invented]', False), ('search[]', False),
    ('search[<keywords>]', False), ('search[x]\nclick[y]', False)])
def test_webshop_parameterized_search(action, allowed):
    model = SimpleNamespace(call=lambda *args: {'action': action})
    history=[{'admissible':['search[<keywords>]', 'click[buy now]']}]
    if allowed: assert actor_call(model, 'goal', history, [], 'test', 0) == action
    else:
        with pytest.raises(ValueError): actor_call(model, 'goal', history, [], 'test', 0)


def test_search_not_available_in_alfworld():
    with pytest.raises(ValueError):
        actor_call(SimpleNamespace(call=lambda *a: {'action':'search[shoes]'}),
                   '', [{'admissible':['look']}], [], 'test', 0)


def test_webshop_terminal_hides_gold(tmp_path):
    env = WebShopEnvironment(str(tmp_path))
    view = env._view('Target secret ASIN and hidden answer', 0.75, True)
    assert view.done and view.score == 0.75 and not view.admissible
    assert 'secret' not in view.observation


def test_webshop_duplicate_goal_rejected():
    with pytest.raises(ValueError, match='duplicate WebShop'):
        validate_tasks([Task('a','a','learn','goal',{'goal_index':4}),
                        Task('b','b','test','goal',{'goal_index':4})])


def test_webshop_reset_explicitly_sets_goal_and_clears_old_session(tmp_path, monkeypatch):
    server=SimpleNamespace(goals=[{},{}],user_sessions={'stale':{'done':True}})
    class FakeEnv:
        def __init__(self, **kwargs):
            self.server=kwargs['server'];self.session='random-constructor-session'
            self.server.user_sessions[self.session]={'done':False}
        def reset(self,session):
            assert not self.server.user_sessions
            self.session=session;self.server.user_sessions[session]={'done':False}
            return f'public goal {session}',None
        def get_available_actions(self):return {'has_search_bar':True,'clickables':['search']}
        def close(self):pass
    monkeypatch.setitem(sys.modules,'web_agent_site.envs.web_agent_text_env',SimpleNamespace(WebAgentTextEnv=FakeEnv))
    env=WebShopEnvironment(str(tmp_path));monkeypatch.setattr(env,'server',lambda:server)
    task=Task('t','g','learn','shop',{'goal_index':1})
    first=env.reset(task);server.user_sessions[1]['done']=True
    assert env.reset(task)==first
    assert server.user_sessions=={1:{'done':False}}
    assert first.observation=='public goal 1'
