from types import SimpleNamespace as S
from evoscope.local_expanded import stratified_test_summary


def test_repeated_trials_do_not_inflate_group_weight():
    tasks=[S(id=str(i),group_id=g,payload={'evaluation_partition':p}) for i,(g,p) in enumerate([('a','unseen'),('a','unseen'),('b','unseen'),('c','seen')])]
    episodes=[S(task_id=str(i),score=score,error=None) for i,score in enumerate([1,1,0,1])]
    result=stratified_test_summary({'frozen':episodes},tasks)['frozen']
    assert result['unseen']['episodes']==3
    assert result['unseen']['independent_groups']==2
    assert result['unseen']['group_macro_success_rate_errors_as_failures']==.5
    assert result['seen']['independent_groups']==1
