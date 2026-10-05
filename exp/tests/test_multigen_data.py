import copy
import json
from pathlib import Path
import pytest
import core
from multigen_data import common as C, collect as G, processing as P

class FakeGenerator:
    def __init__(self):
        self.calls=[]
    def generate(self,messages,**kw):
        self.calls.append((copy.deepcopy(messages),kw))
        text='Initial answer.' if kw['phase']=='initial' else 'Changed answer '+str(kw['slot'])+'.'
        r={'text':text,'finish_reason':'stop',**kw,'generator':'fake','request_id':str(kw['seed']),
           'request_hash':C.digest([messages,kw]),'raw_record_location':'requests/example.json',
           'model_version':'fake-v1','protocol_version':C.SCHEMA}
        r['record_hash']=C.digest(r)
        return r

class Sizer:
    def length(self,row): return 20

def row(split='train'):
    return {'source':'Some source.','source_id':'coedit_gec/source','split':split,'position':0,
            'cluster_id':'source_cluster','reference':'SHOULD NEVER APPEAR'}

def test_initial_has_no_retrieval_and_candidates_share_initial_and_prompt():
    gen=FakeGenerator()
    calls=[]
    def retrieve(source,current):
        assert len(gen.calls)==1
        assert current=='Initial answer.'
        calls.append((source,current))
        return {'text':'FROZEN MEMORY','ids':['one']}
    rec=G.collect_one(row(),'coedit_gec','fake',
       {'train_temperatures':[.1,.1,.7,.7],'nontrain_temperatures':[.1]*4},gen,retrieve)
    assert len(calls)==1 and len(gen.calls)==5
    assert 'FROZEN MEMORY' not in json.dumps(gen.calls[0])
    assert all('FROZEN MEMORY' in json.dumps(c) for c in gen.calls[1:])
    assert len({C.digest(c[0]) for c in gen.calls[1:]})==1
    assert len({c[1]['seed'] for c in gen.calls[1:]})==4
    assert [c[1]['temperature'] for c in gen.calls[1:]]==[.1,.1,.7,.7]
    assert 'SHOULD NEVER APPEAR' not in json.dumps(gen.calls)
    for messages,kw in gen.calls[1:]:
        assert 'Current Draft: Initial answer.' in messages[1]['content']
        assert 'Changed answer' not in messages[1]['content']

def test_reference_change_does_not_change_requests():
    traces=[]
    for reference in ('gold A','gold B'):
        r=row();r['reference']=reference
        gen=FakeGenerator()
        G.collect_one(r,'coedit_gec','fake',
          {'train_temperatures':[.1,.1,.7,.7],'nontrain_temperatures':[.1]*4},
          gen,lambda *a:{'text':'memory','ids':['m']})
        traces.append(gen.calls)
    assert traces[0]==traces[1]

def test_seeds_are_deterministic_distinct_and_model_specific():
    a=C.candidate_seeds('gigaword','train','source','a')
    assert len(set(a))==4
    assert a==C.candidate_seeds('gigaword','train','source','a')
    assert a!=C.candidate_seeds('gigaword','train','source','b')

def test_input_has_exactly_three_fields_and_no_task_prefix():
    x={'source':'source','current':'before','candidate':'after','task':'SECRET','generator':'SECRET'}
    assert list(C.model_input(x))==['source','current','candidate']
    assert C.state_text(x)=='Source:\nsource\n\nCurrent:\nbefore\n\nCandidate:\nafter'

def test_english_is_valid_and_score_ties_are_reject():
    assert C.valid_pair('coedit_gec','One sentence.',{'text':'Another one.','finish_reason':'stop'})[0]
    assert not C.valid_pair('wmt19_en_zh','句子',{'text':'English','finish_reason':'stop'})[0]
    assert C.label_scores(.5,.5)==('Reject',0)
    assert C.label_scores(.5,.50000001)[0]=='Accept'
    with pytest.raises(ValueError):C.label_scores(float('nan'),.5)

@pytest.mark.parametrize('entry',['clean','score','export','evaluation_rows'])
def test_sealed_test_rejected_before_reading_any_data(tmp_path,entry):
    with pytest.raises(PermissionError,match='sealed'):
        getattr(P,entry)(tmp_path,'test')
    assert not list(tmp_path.iterdir())

def test_incomplete_lock_cannot_unseal(tmp_path):
    C.dump(tmp_path/'evaluation_lock.json',{'schema':C.SCHEMA})
    with pytest.raises(PermissionError):C.require_unsealed(tmp_path,'test')

def make_raw(directory,gen,candidates):
    fake=FakeGenerator()
    rec=G.collect_one(row(),'coedit_gec',gen,
        {'train_temperatures':[.1,.1,.7,.7],'nontrain_temperatures':[.1]*4},
        fake,lambda *a:{'text':'memory','ids':['m']})
    for i,candidate in enumerate(rec['candidates']):
        candidate['text']=candidates[i]
        candidate['generator']=gen
        candidate['request_hash']=C.digest([gen,i])
        candidate['request_id']=candidate['request_hash']
        candidate['record_hash']=C.digest({k:v for k,v in candidate.items() if k!='record_hash'})
    rec['record_hash']=C.digest(rec)
    C.dump(directory/'raw/train'/gen/'ab/source.json',rec,immutable=True)
    return rec

def test_dedup_keeps_all_provenance_noops_audited_and_resume_idempotent(tmp_path):
    C.dump(tmp_path/'protocol.json',{'task':'coedit_gec','feedback':{'metric':'nltk_sentence_gleu'}})
    choices=['Changed.','Initial answer.','Changed.','Changed but tie.']
    make_raw(tmp_path,'model_a',choices)
    make_raw(tmp_path,'model_b',choices)
    result=P.clean(tmp_path,'train',Sizer())
    assert result['raw_pairs']==8
    assert result['unique_valid_changed_pairs']==2
    assert result['no_op_pairs']==2
    assert result['duplicate_valid_changed_occurrences']==4
    assert P.clean(tmp_path,'train',Sizer())['raw_pairs']==8
    db=P.database(tmp_path)
    db.execute("UPDATE pairs SET label='Reject',q_current=.5,q_candidate=.5,delta=0")
    db.commit();db.close()
    out=P.export(tmp_path,'train')
    exported=list(C.jsonl(out['path']))
    assert len(exported)==2
    assert sorted(len(x['metadata']['provenance']) for x in exported)==[2,4]
    assert all(x['label']=='Reject' for x in exported)
    required={'generator','temperature','slot','seed','request_hash','raw_record_location'}
    assert all(required<=set(p) for x in exported for p in x['metadata']['provenance'])
    assert len({p['generator'] for p in exported[0]['metadata']['provenance']})==2

def test_sealed_audit_never_compares_answers_or_reports_quality(tmp_path,monkeypatch):
    r=make_raw(tmp_path,'m',['Initial answer.']*4)
    r['split']='test'
    r['record_hash']=C.digest({k:v for k,v in r.items() if k!='record_hash'})
    C.dump(tmp_path/'raw/test/m/ab/source.json',r,immutable=True)
    monkeypatch.setattr(C,'valid_pair',lambda *a:pytest.fail('quality cleaning called'))
    monkeypatch.setattr(C,'label_scores',lambda *a:pytest.fail('scoring called'))
    audit=G.sealed_audit(tmp_path)
    assert audit['checks']['requests']==5
    assert not any(k in json.dumps(audit) for k in ('delta','Accept','Reject','noop','changed'))

def test_source_bootstrap_keeps_all_generator_pairs_together():
    import numpy as np
    rows=[{'label':label,'metadata':{'cluster_id':sid,'delta':0.1}} for sid,label in
            [('s1','Accept'),('s1','Accept'),('s2','Reject')]]
    pred=['Accept']*3
    result=P.clustered_bootstrap(rows,{'a':pred,'b':pred},draws=2000)
    np.testing.assert_array_equal(result['a']['accuracy'],result['b']['accuracy'])
    # Sampling the two sources gives only all-s1, one of each, or all-s2.
    assert set(result['a']['accuracy'])=={0.,2/3,1.}

def test_parallel_timing_is_union_not_sum():
    assert C.union_seconds([(0,10),(1,9),(8,12),(20,23)])==15

def test_immutable_records_cannot_be_overwritten(tmp_path):
    p=tmp_path/'one.json'
    C.dump(p,{'x':1},immutable=True);C.dump(p,{'x':1},immutable=True)
    with pytest.raises(ValueError):C.dump(p,{'x':2},immutable=True)


def test_legacy_evaluator_cannot_read_sealed_test(tmp_path):
    import laya_acceptance_common as legacy
    C.dump(tmp_path/'test_seal.json', {'state':'sealed'})
    p=tmp_path/'exports/test.jsonl'
    with pytest.raises(PermissionError,match='sealed'):
        legacy.read_jsonl(p)

def test_unseal_requires_frozen_artifacts_and_detects_modified_weights(tmp_path):
    C.dump(tmp_path/'protocol.json',{'schema':C.SCHEMA})
    weights=tmp_path/'weights.safetensors';weights.write_bytes(b'weights')
    specfile=tmp_path/'training_spec.json';C.dump(specfile,{'class_weights':None})
    art={str(specfile):C.file_hash(specfile)}
    lock={'model':{'checkpoint_weights':str(weights),'artifacts':{str(weights):C.file_hash(weights)}},
          'loss':{'statistics_split':'train','artifacts':art},
          'selection_rule':{'uses_test':False,'selection_split':'dev','artifacts':art},
          'evaluation':{'bootstrap_unit':'source','bootstrap_draws':2000,'comparisons':['frozen_model'],'artifacts':art}}
    C.freeze_evaluation_lock(tmp_path,lock)
    C.require_unsealed(tmp_path,'test')
    weights.write_bytes(b'changed')
    with pytest.raises(PermissionError):C.require_unsealed(tmp_path,'test')


def test_old_nul_padded_records_are_read_without_changing_original(tmp_path):
    from multigen_data.prepare import historical_jsonl,history_hashes
    p=tmp_path/'old.jsonl'
    original=chr(0)*12+C.canonical({'source':'old source','source_hash':'a'*64})+'\n'
    p.write_text(original)
    repairs=[];rows=list(historical_jsonl(p,repairs))
    assert rows[0]['source']=='old source'
    assert list(history_hashes(rows[0]))==['a'*64]
    assert repairs==[{'line':1,'nul_bytes_removed':12}]
    assert p.read_text()==original


def test_real_laya_encoding_uses_new_three_field_template_and_legacy_stays_compatible():
    import laya_acceptance_common as legacy
    tok=legacy.get_tokenizer()
    inputs={'source':'A source','current':'Old text','candidate':'New text'}
    rows=[{'id':'p','input':inputs,'label':'Accept','metadata':{'input_template_version':C.INPUT_VERSION}}]
    encoded=legacy.EncodedPairs(rows,tok)
    assert encoded.template_version==C.INPUT_VERSION
    decoded=tok.decode(encoded.states[0][0])
    assert 'Task:' not in decoded and 'English-to-Chinese' not in decoded
    assert 'Task: English-to-Chinese Translation' in legacy.state_text(inputs)
    assert 'Task:' not in legacy.state_text(inputs,template_version=C.INPUT_VERSION)
    with pytest.raises(ValueError):
        legacy.EncodedPairs(rows+[{'id':'old','input':inputs,'label':'Accept'}],tok)
