import json
import threading
from pathlib import Path
import pytest
from multigen_data import common as C, processing as P
import multigen_streaming as S
import run_multigen_pipeline as R
from run_multigen_async import shards, common_prefix
from test_multigen_data import make_raw, Sizer


def entries(d):
    return [{'path':str(p),'sha256':C.file_hash(p)} for p in sorted((d/'raw/train').glob('*/*/*.json'))]


def tables(d):
    db=P.database(d)
    result={t:db.execute('SELECT * FROM '+t+' ORDER BY 1').fetchall() for t in ['pairs','occurrences','processed']}
    # Two distinct temporary roots legitimately differ in processed absolute paths.
    result['processed']=[(str(Path(p).relative_to(d)),h) for p,h in result['processed']]
    db.close()
    return result


def fixture(d):
    C.dump(d/'protocol.json',{'task':'coedit_gec','feedback':{'metric':'nltk_sentence_gleu'}})
    choices=['Changed.','Initial answer.','Changed.','Changed but tie.']
    for gen in ['model_a','model_b']:
        make_raw(d,gen,choices)


def test_incremental_matches_frozen_cleaner_and_preserves_all_provenance(tmp_path):
    old,new=tmp_path/'old',tmp_path/'new'
    fixture(old);fixture(new)
    report=P.clean(old,'train',Sizer())
    e=entries(new)
    assert S.clean_entries(new,'train',e[:1],Sizer())['unique_valid_changed_pairs']==2
    current=S.clean_entries(new,'train',e[1:],Sizer())
    assert current['raw_pairs']==report['raw_pairs']==8
    assert tables(old)==tables(new)
    db=P.database(new)
    db.execute("UPDATE pairs SET label='Reject',q_current=.5,q_candidate=.5,delta=0")
    db.commit();db.close()
    out=P.export(new,'train')
    rows=list(C.jsonl(out['path']))
    assert len(rows)==2
    assert sorted(len(r['metadata']['provenance']) for r in rows)==[2,4]
    assert all(r['label']=='Reject' and r['input']['current']!=r['input']['candidate'] for r in rows)


def test_resume_skips_old_raw_reads_without_double_counting(tmp_path,monkeypatch):
    fixture(tmp_path);e=entries(tmp_path)
    S.clean_entries(tmp_path,'train',e,Sizer())
    original=Path.read_bytes
    def forbidden(path):
        if '/raw/' in str(path):pytest.fail('Previously processed raw reread')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    result=S.clean_entries(tmp_path,'train',e,Sizer())
    assert result['raw_pairs']==8 and result['new_raw_files']==0 and result['reused_raw_files']==2
    broken=[dict(e[0],sha256='incorrect')]
    with pytest.raises(ValueError,match='conflicts'):
        S.clean_entries(tmp_path,'train',broken,Sizer())


def test_failed_batch_rolls_back_and_recovers_exactly(tmp_path):
    fixture(tmp_path);e=entries(tmp_path)
    corrupt=[e[0],dict(e[1],sha256='broken')]
    with pytest.raises(ValueError,match='immutable'):
        S.clean_entries(tmp_path,'train',corrupt,Sizer(),batch_size=128)
    assert not tables(tmp_path)['processed']
    assert S.clean_entries(tmp_path,'train',e,Sizer())['raw_pairs']==8


def test_final_integrity_audit_detects_old_file_mutation(tmp_path):
    fixture(tmp_path);e=entries(tmp_path)
    S.clean_entries(tmp_path,'train',e,Sizer())
    for item in e:
        model=Path(item['path']).parts[-3]
        C.dump(tmp_path/'completed/train'/model/'00000000-00000001.json',{'raw_records':[item]})
    assert S.audit_processed(tmp_path,'train')['verified_raw_files']==2
    p=Path(e[0]['path']);p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='integrity'):
        S.audit_processed(tmp_path,'train')


@pytest.mark.parametrize('entry',['clean','audit','manifest'])
def test_sealed_test_blocked_before_io(tmp_path,entry):
    def forbidden():
        pytest.fail('Test iterator read')
        yield
    with pytest.raises(PermissionError):
        if entry=='clean':S.clean_entries(tmp_path,'test',forbidden(),Sizer())
        elif entry=='audit':S.audit_processed(tmp_path,'test')
        else:list(S.published_entries(tmp_path,'test'))
    assert not list(tmp_path.iterdir())


class FakeCollector:
    def __init__(self,d,name):
        self.published_directory=d;self.model=name;self.ready_set=set();self.calls=[]
    def ready(self,split,start,stop):return (split,start,stop) in self.ready_set
    def run(self,split,start,stop):
        self.calls.append((split,start,stop));self.ready_set.add((split,start,stop))
    def publish(self,split,start,stop):
        result={'sources':stop-start,'raw_records':[]}
        C.dump(self.published_directory/'completed'/split/self.model/f'{start:08d}-{stop:08d}.json',result,immutable=True)
        return result


def state(path,task='a',target=16):
    protocol=dict(task=task,pool_counts={'train':136},initial_train_sources=132,
                  shard_sources=2,target_unique_valid_changed=target,candidates=4)
    C.dump(path/'protocol.json',protocol)
    collectors=[FakeCollector(path,'fast'),FakeCollector(path,'slow')]
    return R.TaskState(path,protocol,collectors)


def test_ready_queue_uses_holdouts_and_next_task_without_waiting_for_cpu(tmp_path):
    a=state(tmp_path/'a');b=state(tmp_path/'b','b')
    # First task has exhausted its currently authorized train prefix, but CPU
    # has not counted it; ready holdouts/other task must still use the model.
    for c in a.collectors.values():
        for start,stop in a.ranges:
            if stop<=a.limit:c.run('train',start,stop)
    h=R.Hub([a,b])
    selected=[]
    for _ in range(3):
        s,job=h.claim('fast');selected.append((s.task,job[0]));s.collectors['fast'].run(*job)
    assert selected==[('a','dev'),('a','test'),('b','train')]
    assert all(stop<=a.limit for split,start,stop in a.collectors['fast'].calls if split=='train')


def test_generation_advances_while_cpu_cleaning_is_blocked_and_stops_at_exact_target(tmp_path,monkeypatch):
    a=state(tmp_path/'a');b=state(tmp_path/'b','b')
    h=R.Hub([a,b]);threads=[];cleaning=threading.Event();advanced=threading.Event()
    calls=[]
    def clean(d,split,entries,sizer):
        assert split=='train'
        list(entries)
        if not calls:
            cleaning.set()
            assert advanced.wait(3),'GPU work waited for CPU cleaning'
        calls.append(True)
        f=common_prefix(d,['fast','slow'],shards(136,132,2))
        return {'unique_valid_changed_pairs':(f-128)*4,'raw_pairs':(f-128)*8}
    monkeypatch.setattr(R,'clean_entries',clean)
    def generator(model):
        while (claim:=h.claim(model)) is not None:
            s,job=claim
            if s is b:
                assert cleaning.wait(3)
            s.collectors[model].run(*job);h.notify()
            if s is b:advanced.set()
    for model in a.collectors:
        t=threading.Thread(target=generator,args=(model,),daemon=True);t.start();threads.append(t)
    try:
        result=R.train_incremental(a,h,Sizer())
        assert result['target_met'] and result['train_sources']==132
        for c in a.collectors.values():
            assert [x for x in c.calls if x[0]=='train']==[('train',128,130),('train',130,132)]
        assert R.train_incremental(a,h,Sizer())==result
    finally:
        h.stop()
        for t in threads:t.join(timeout=3)
    assert all(not t.is_alive() for t in threads)
