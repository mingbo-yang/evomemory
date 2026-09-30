"""Separate probability/threshold calibration, then a once-frozen held-out test."""
from datetime import datetime,timezone
import math
import time
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from laya_acceptance_common import *


def fit_temperature(z,rows):
    y=np.array([LABELS.index(r['label']) for r in rows])
    def nll(log_t):
        p=probabilities(z,np.exp(log_t))
        return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())
    fit=minimize_scalar(nll,bounds=(np.log(.1),np.log(10.)),method='bounded',options={'xatol':1e-7})
    assert fit.success
    candidates=[float(fit.x),0.,np.log(.1),np.log(10.)]
    log_t=min(candidates,key=nll)
    return {'temperature':float(np.exp(log_t)),'nll_before':nll(0.),'nll_after':nll(log_t)}


def select_threshold(rows,z,delta,temperature,protocol):
    p=probabilities(z,temperature);grid=[];best=None;best_key=None
    for tb in protocol['p_better_grid']:
        for tw in protocol['p_worse_max_grid']:
            threshold={'p_better':tb,'p_worse_max':tw}
            stat=acceptance_stats(rows,p,delta,threshold)
            grid.append({'threshold':threshold,**stat})
            if stat['coverage']>=.10 and stat['precision'] is not None and stat['precision']>=.80 and stat['mean_delta']>0:
                key=(stat['coverage'],stat['precision'],stat['mean_delta'],tb,-tw)
                if best_key is None or key>best_key:best=grid[-1];best_key=key
    return {'threshold':None if best is None else best['threshold'],
            'status':'PASS' if best else 'NO_FEASIBLE_THRESHOLD', 'selected':best,'grid':grid}


def bootstrap_acceptance(rows,p,delta,sources,threshold,replicates=2000):
    mask=accepted_mask(rows,p,threshold);delta=np.asarray(delta)
    names=sorted(set(sources));positions={s:i for i,s in enumerate(names)}
    totals=np.zeros((len(names),4))
    for i,s in enumerate(sources):
        j=positions[s];totals[j,0]+=1
        if mask[i]:totals[j,1]+=1;totals[j,2]+=delta[i]>.01;totals[j,3]+=delta[i]
    rng=np.random.default_rng(20260926)
    vals={'coverage':[],'precision':[],'mean_delta':[]}
    for _ in range(replicates):
        a=totals[rng.integers(len(names),size=len(names))].sum(0)
        vals['coverage'].append(a[1]/a[0])
        if a[1]:vals['precision'].append(a[2]/a[1]);vals['mean_delta'].append(a[3]/a[1])
    return {'unit':'source cluster','replicates':replicates,'source_count':len(names),
            'ci95':{k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in vals.items()},
            'replicates_with_acceptance':len(vals['precision'])}


def gate(stats,boot):
    lower=boot['ci95']['mean_delta']
    checks={'nonzero_acceptance':stats['accepted']>0,'coverage':stats['coverage']>=.10,
            'precision':stats['precision'] is not None and stats['precision']>=.80,
            'mean_gain':stats['mean_delta'] is not None and stats['mean_delta']>0,
            'gain_ci_lower_positive':lower is not None and lower[0]>0}
    return {'pass':all(checks.values()),'checks':checks}


def unseal_test(lock):
    # Executed only after every checkpoint, temperature and threshold was frozen.
    assert (OUT/'evaluation_lock.json').exists()
    assert json.loads((OUT/'evaluation_lock.json').read_text())==lock
    import acceptance_data as collection
    from core.scoring import Scorer
    from prepare_acceptance_training import convert
    refs={r.sample_id:r for r in collection.read_manifest(COLLECTION/'test_manifest.jsonl')}
    raw=[r for r in collection.unique_rows(collection.iter_raw('test')) if r['valid']]
    scorer=Scorer('wmt19_en_zh');rows=[];delta=[];sources=[]
    for r in raw:
        ref=refs[r['sample_id']]
        d=(scorer.primary(ref.reference,r['candidate'])-scorer.primary(ref.reference,r['before']))/100
        r['delta']=d
        rows.append(convert(r,.01));delta.append(d);sources.append(r['sample_id'])
    assert len(rows)==793
    path=OUT/'evaluation/test_labels.jsonl';path.parent.mkdir(exist_ok=True)
    path.write_text(''.join(json.dumps({'id':r['id'],'label':r['label'],'delta':d,'sample_id':s},ensure_ascii=False)+'\n' for r,d,s in zip(rows,delta,sources)))
    return rows,delta,sources


def runtime_parity(rows,z,temperature):
    # This is a serialization/API test, not a new evaluation or threshold search.
    import laya
    states=[state_text(r['input']) for r in rows[:8]]
    agent=laya.Agent(str(OUT/'best'),device='cuda',compile=False,fast=False)
    questions={'acceptance':{'type':'choice','instructions':QUESTION['ins'],'criteria':CRITERIA}}
    results=agent.predict_batch(states,questions)
    api=np.array([[r['answers']['acceptance']['probabilities'][label] for label in LABELS] for r in results])
    direct=probabilities(z[:8],temperature)
    error=float(abs(api-direct).max())
    result={'max_absolute_probability_difference':error,'tolerance':.02,'passed':error<=.02}
    dump(OUT/'runtime_parity.json',result)
    assert result['passed'], 'Saved official Agent probabilities differ from evaluated model'
    del agent;torch.cuda.empty_cache()


def run():
    torch.set_num_threads(4)
    protocol=json.loads((OUT/'training_protocol.json').read_text())
    selection=json.loads((OUT/'best_selection.json').read_text())
    assert digest(OUT/'best_uncalibrated/model.safetensors')==selection['checkpoint_sha256']
    tok=get_tokenizer()
    cal,c_delta,c_source=load_fold('temperature_calibration')
    thr,t_delta,t_source=load_fold('threshold_calibration')
    ce=EncodedPairs(cal,tok);te=EncodedPairs(thr,tok)
    params={}
    for name,path in [('base',BASE),('finetuned',OUT/'best_uncalibrated')]:
        model,cfg=load_model(path);model.to('cuda')
        zc=infer(model,ce);zt=infer(model,te)
        fitted=fit_temperature(zc,cal)
        operating=select_threshold(thr,zt,t_delta,fitted['temperature'],protocol)
        params[name]={'model_sha256':digest(path/'model.safetensors'),**fitted,
                      'threshold':operating['threshold'],'threshold_status':operating['status']}
        np.save(OUT/f'{name}_temperature_calibration_logits.npy',zc)
        np.save(OUT/f'{name}_threshold_calibration_logits.npy',zt)
        dump(OUT/f'{name}_threshold_search.json',operating)
        if name=='finetuned':
            save_checkpoint(model,cfg,tok,OUT/'best',temperature=fitted['temperature'])
            runtime_parity(cal,zc,fitted['temperature'])
        del model;torch.cuda.empty_cache()
    lock={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'selected_epoch':selection['epoch'],
          'models':params,'protocol_sha256':digest(OUT/'training_protocol.json'),
          'dataset_sha256':digest(DATA/'dataset.json'),'final_checkpoint_sha256':digest(OUT/'best/model.safetensors'),
          'test_manifest_sha256':digest(COLLECTION/'test_manifest.jsonl'),
          'evaluator_sha256':digest(Path(__file__))}
    assert not (OUT/'evaluation_lock.json').exists(),'Evaluation is already frozen; do not retune against test'
    dump(OUT/'evaluation_lock.json',lock)
    dump(OUT/'status.json',{'stage':'heldout_evaluation','updated_at_utc':datetime.now(timezone.utc).isoformat()})
    test,delta,sources=unseal_test(lock);encoded=EncodedPairs(test,tok)
    report={'selected_epoch':selection['epoch'],'label_epsilon':.01,'feedback_semantics':'weak sentence feedback, not semantic gold',
            'test_sources':len(set(sources)),'test_pairs':len(test),'models':{},'calibration':params}
    for name,path in [('base',BASE),('finetuned',OUT/'best')]:
        model,cfg=load_model(path);model.to('cuda');start=time.perf_counter()
        z=infer(model,encoded);torch.cuda.synchronize();elapsed=time.perf_counter()-start
        np.save(OUT/'evaluation'/f'{name}_test_logits.npy',z)
        temp=params[name]['temperature'];threshold=params[name]['threshold']
        result=metrics(test,z,delta,temp,threshold)
        p=probabilities(z,temp)
        boot=bootstrap_acceptance(test,p,delta,sources,threshold)
        result.update(bootstrap=boot,gate=gate(result['calibrated_acceptance'],boot),
                      uncalibrated=metrics(test,z,delta,1.0),batch_inference_s=elapsed,
                      amortized_ms_per_pair=elapsed/len(test)*1000)
        # Post-freeze presentation diagnostic; it cannot select a checkpoint/threshold.
        reverse=infer(model,encoded,force_order=[2,1,0])
        reverse_p=probabilities(reverse,temp)
        result['option_order_diagnostic']={'semantic_argmax_agreement':float(np.mean(p.argmax(1)==reverse_p.argmax(1))),
                                           'mean_abs_probability_change':float(abs(p-reverse_p).mean())}
        report['models'][name]=result
        del model;torch.cuda.empty_cache()
    report['verifier_gate_passed']=report['models']['finetuned']['gate']['pass']
    report['next_stage']='static_pipeline_required' if report['verifier_gate_passed'] else 'verifier_failed_do_not_run_main_experiments'
    dump(OUT/'evaluation_report.json',report)
    lines=['# Laya multilingual acceptance: training and validation','',
           f"Base: convaiinnovations/laya-multilingual; selected epoch {report['selected_epoch']}; test {len(test)} pairs from {len(set(sources))} sources.",
           '', 'Training labels use the original weak-feedback proxy, epsilon 0.01. This evaluation measures that proxy objective, not independently annotated semantic correctness.',
           '', '| Model | Better AUC (changed) | Macro F1 | Brier | ECE | Accepted | Coverage | Precision | Mean delta |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return 'undefined' if v is None else f'{v:.4f}'
    for name,m in report['models'].items():
        a=m['calibrated_acceptance']
        lines.append(f"| {name} | {fmt(m['auc_better_changed'])} | {fmt(m['macro_f1'])} | {fmt(m['brier'])} | {fmt(m['ece_15'])} | {a['accepted']} | {fmt(a['coverage'])} | {fmt(a['precision'])} | {fmt(a['mean_delta'])} |")
    lines+=['',f"Verifier gate passed: {report['verifier_gate_passed']}. Next stage: {report['next_stage']}.",
            '', 'Temperature fitting and operating-point selection used separate calibration folds. No feasible threshold means acceptance disabled and a failed gate, not successful accuracy. Default pBetter >= 0.5 results, raw scores, source-bootstrap confidence intervals, exact thresholds and option-order diagnostics are retained in evaluation_report.json.',
            '', 'No online memory experiment is justified by classification metrics alone. If this verifier gate passes, a real fixed-memory rollout remains required before full_static/full_online comparison.',
            '', 'Official training source: https://github.com/NandhaKishorM/laya ; base model: https://huggingface.co/convaiinnovations/laya-multilingual .',
            '', f'Checkpoint: {OUT / "best"}']
    text='\n'.join(lines)+'\n'
    (OUT/'EVALUATION.md').write_text(text)
    (ROOT/'reports/LAYA_ACCEPTANCE_VALIDATION.md').write_text(text)
    dump(OUT/'status.json',{'stage':'verifier_validation_complete','updated_at_utc':datetime.now(timezone.utc).isoformat(),
                           'verifier_gate_passed':report['verifier_gate_passed'],'next_stage':report['next_stage']})
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':run()
