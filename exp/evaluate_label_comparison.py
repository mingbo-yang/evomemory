"""Freeze calibration before opening pilot test labels; compare two label sources."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score
import laya_acceptance_common as C
from semantic_label_common import OUT,ROOT,LABELS,dump,read_jsonl
from train_label_comparison import ready,read_fold,resolved,utility,save


def status(stage,**kw):
    dump(OUT/'comparison_status.json',{'stage':stage,'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw})


def fit_temperature(rows,z):
    mask=np.array([r['semantic_label']!='Uncertain' for r in rows])
    y=np.array([LABELS.index(r['semantic_label']) for r in rows if r['semantic_label']!='Uncertain'])
    assert len(y)>=10
    z=z[mask]
    def loss(t):
        p=C.probabilities(z,np.exp(t))
        return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())
    opt=minimize_scalar(loss,bounds=(np.log(.1),np.log(10)),method='bounded')
    assert opt.success
    t=min([float(opt.x),0.,np.log(.1),np.log(10)],key=loss)
    return {'temperature':float(np.exp(t)),'resolved_n':len(y),'nll_before':loss(0),'nll_after':loss(t)}


def wilson(k,n):
    if n==0:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,mid-width)),float(min(1,mid+width))]


def admission(rows,p,threshold=None,mask=None):
    if mask is None:mask=C.accepted_mask(rows,p,threshold)
    count=Counter(r['semantic_label'] for r,m in zip(rows,mask) if m)
    n=int(mask.sum());better=count['Better'];worse=count['Worse'];u=count['Uncertain']
    return {'accepted':n,'coverage':n/len(rows),'counts':dict(count),
            'confirmed_precision_lower':better/n if n else None,
            'possible_precision_upper':(better+u)/n if n else None,
            'confirmed_precision_wilson95':wilson(better,n),
            'worst_case_utility_per_accepted':(better-worse-u)/n if n else None,
            'worst_case_utility_per_pair':(better-worse-u)/len(rows)}


def select_threshold(rows,z,temp,protocol):
    p=C.probabilities(z,temp);grid=[];best=None;best_key=None
    for tb in protocol['p_better_grid']:
        for tw in protocol['p_worse_max_grid']:
            threshold={'p_better':tb,'p_worse_max':tw}
            stats=admission(rows,p,threshold);record={'threshold':threshold,**stats};grid.append(record)
            if stats['coverage']>=.1 and (stats['confirmed_precision_lower'] or 0)>=.8 and stats['worst_case_utility_per_pair']>0:
                key=(stats['coverage'],stats['confirmed_precision_lower'],stats['worst_case_utility_per_pair'],tb,-tw)
                if best_key is None or key>best_key:best=record;best_key=key
    return {'status':'PASS' if best else 'NO_FEASIBLE_THRESHOLD',
            'threshold':best['threshold'] if best else None,'selected':best,'grid':grid}


def classification(rows,z,temp,label_key='semantic_label'):
    mask=np.array([r[label_key]!='Uncertain' for r in rows])
    selected=[{'id':r['id'],'input':r['input'],'label':r[label_key]} for r in rows if r[label_key]!='Uncertain']
    if not selected:return {'n':0}
    m=C.metrics(selected,z[mask],utility(selected),temp)
    for key in ['default_acceptance','calibrated_acceptance','top10pct']:m.pop(key)
    y=np.array([LABELS.index(r['label']) for r in selected]);pred=C.probabilities(z[mask],temp).argmax(1)
    m['label_counts']=dict(Counter(r['label'] for r in selected))
    m['prediction_counts']=dict(Counter(LABELS[i] for i in pred))
    m['confusion_matrix_true_rows_pred_columns']=confusion_matrix(y,pred,labels=[0,1,2]).tolist()
    m['better_precision']=float(np.mean(y[pred==0]==0)) if (pred==0).any() else None
    m['better_recall']=float(np.mean(pred[y==0]==0)) if (y==0).any() else None
    return m


def paired_bootstrap(rows,logits,params,replicates=2000):
    # Sampling unit is source: the frozen pilot contains exactly one pair/source.
    assert len(set(' '.join(r['input']['source'].casefold().split()) for r in rows))==len(rows)
    y=np.array([LABELS.index(r['semantic_label']) if r['semantic_label']!='Uncertain' else -1 for r in rows])
    p={k:C.probabilities(v,params[k]['temperature']) for k,v in logits.items() if k in ['bleu','semantic']}
    values={'accuracy':[],'macro_f1':[],'better_auc':[]};rng=np.random.default_rng(20260926)
    for _ in range(replicates):
        idx=rng.integers(len(rows),size=len(rows));idx=idx[y[idx]>=0];yy=y[idx]
        if not len(idx):continue
        d={}
        for arm in p:
            pp=p[arm][idx];pred=pp.argmax(1)
            d[arm]={'accuracy':float((pred==yy).mean()),
                    'macro_f1':float(f1_score(yy,pred,labels=[0,1,2],average='macro',zero_division=0))}
            if len(set(yy==0))==2:d[arm]['better_auc']=float(roc_auc_score(yy==0,pp[:,0]))
        for key in values:
            if key in d['bleu'] and key in d['semantic']:values[key].append(d['semantic'][key]-d['bleu'][key])
    return {'unit':'source, one pair per source','replicates':replicates,'difference':'semantic minus BLEU',
            'ci95':{k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in values.items()},
            'valid_replicates':{k:len(v) for k,v in values.items()}}



def format_diagnostic(test,logits,params):
    assert (OUT/'comparison_evaluation_lock.json').exists()
    result={'role':'secondary only, same frozen predictions and thresholds; no tuning','folds':{},'models':{}}
    for fold in ['development','temperature_calibration','threshold_calibration','test']:
        primary=read_fold(fold,allow_test=fold=='test')
        labels={r['id']:r['label'] for r in read_jsonl(OUT/'labels_compact'/f'{fold}.jsonl')}
        both=[r for r in primary if r['semantic_label']!='Uncertain' and labels[r['id']]!='Uncertain']
        result['folds'][fold]={'n':len(primary),'agreement_all':sum(r['semantic_label']==labels[r['id']] for r in primary)/len(primary),
              'both_resolved':len(both),'agreement_both_resolved':sum(r['semantic_label']==labels[r['id']] for r in both)/len(both) if both else None,
              'primary_counts':dict(Counter(r['semantic_label'] for r in primary)),
              'compact_counts':dict(Counter(labels.values()))}
        if fold=='test':changed=[dict(r,semantic_label=labels[r['id']]) for r in test]
    for arm,z in logits.items():
        temp=params[arm]['temperature'];p=C.probabilities(z,temp)
        result['models'][arm]={'classification':classification(changed,z,temp),
                             'acceptance_at_primary_frozen_threshold':admission(changed,p,params[arm]['threshold'])}
    return result


def runtime_parity(path,rows,z,temp):
    import laya
    agent=laya.Agent(str(path),device='cuda',compile=False,fast=False)
    question={'acceptance':{'type':'choice','instructions':C.QUESTION['ins'],'criteria':C.CRITERIA}}
    out=agent.predict_batch([C.state_text(r['input']) for r in rows[:8]],question)
    actual=np.array([[r['answers']['acceptance']['probabilities'][k] for k in LABELS] for r in out])
    err=float(abs(actual-C.probabilities(z[:8],temp)).max())
    del agent;torch.cuda.empty_cache()
    assert err<=.02,(path,err)
    return {'max_absolute_probability_error':err,'tolerance':.02,'passed':True}


def run():
    torch.set_num_threads(4)
    while not all((OUT/a/'selection.json').exists() for a in ['bleu','semantic']) or not all(ready(f) for f in ['temperature_calibration','threshold_calibration']):
        status('waiting_for_models_and_calibration_labels');time.sleep(10)
    protocol=json.loads((OUT/'pilot_protocol.json').read_text());tok=C.get_tokenizer()
    paths={'base':C.BASE,**{a:OUT/a/'best_uncalibrated' for a in ['bleu','semantic']}}
    lockpath=OUT/'comparison_evaluation_lock.json'
    if lockpath.exists():
        lock=json.loads(lockpath.read_text());params=lock['models']
        assert lock['protocol_sha256']==C.digest(OUT/'pilot_protocol.json')
        assert lock['training_label_sha256']==C.digest(OUT/'labels/train.jsonl')
        for fold,h in lock['calibration_label_hashes'].items():assert C.digest(OUT/'labels'/f'{fold}.jsonl')==h
        for arm,path in paths.items():assert params[arm]['model_sha256']==C.digest(path/'model.safetensors')
    else:
        status('calibrating_before_test');cal=read_fold('temperature_calibration');thr=read_fold('threshold_calibration')
        ce=C.EncodedPairs(cal,tok);te=C.EncodedPairs(thr,tok);params={}
        for arm,path in paths.items():
            model,cfg=C.load_model(path);model.to('cuda')
            zc=C.infer(model,ce);zt=C.infer(model,te)
            fit=fit_temperature(cal,zc);op=select_threshold(thr,zt,fit['temperature'],protocol)
            params[arm]={'model_sha256':C.digest(path/'model.safetensors'),**fit,
                         'threshold':op['threshold'],'threshold_status':op['status']}
            np.save(OUT/f'{arm}_temperature_logits.npy',zc);np.save(OUT/f'{arm}_threshold_logits.npy',zt)
            dump(OUT/f'{arm}_threshold_search.json',op)
            if arm!='base':
                save(model,cfg,tok,OUT/arm/'best',arm,fit['temperature'])
                params[arm]['saved_model_sha256']=C.digest(OUT/arm/'best/model.safetensors')
                params[arm]['selected_epoch']=json.loads((OUT/arm/'selection.json').read_text())['epoch']
            del model;torch.cuda.empty_cache()
            if arm!='base':params[arm]['runtime_parity']=runtime_parity(OUT/arm/'best',cal,zc,fit['temperature'])
            print('CALIBRATED',arm,params[arm],flush=True)
        lock={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'models':params,
              'protocol_sha256':C.digest(OUT/'pilot_protocol.json'),
              'secondary_format_audit_protocol_sha256':C.digest(OUT/'format_audit_protocol.json'),
              'input_hashes':protocol['input_hashes'],
              'training_label_sha256':C.digest(OUT/'labels/train.jsonl'),
              'calibration_label_hashes':{f:C.digest(OUT/'labels'/f'{f}.jsonl') for f in ['development','temperature_calibration','threshold_calibration']},
              'script_sha256':C.digest(Path(__file__))}
        dump(lockpath,lock)
    while not ready('test'):status('frozen_waiting_for_test_labels');time.sleep(10)
    while not (OUT/'compact_test_status.json').exists() or json.loads((OUT/'compact_test_status.json').read_text()).get('state')!='complete':
        status('frozen_waiting_for_secondary_format_audit');time.sleep(10)
    status('evaluating_frozen_test');test=read_fold('test',allow_test=True);enc=C.EncodedPairs(test,tok)
    labels=Counter(r['semantic_label'] for r in test);known=[r for r in test if r['semantic_label']!='Uncertain']
    report={'scope':'paired small pilot; shared model-generated semantic reference, no human gold or online rollout',
            'test_pairs':len(test),'resolved_pairs':len(known),'test_label_counts':dict(labels),
            'training':json.loads((OUT/'matched_training_summary.json').read_text()),
            'semantic_vs_bleu_agreement_on_resolved_test':sum(r['semantic_label']==r['bleu_label'] for r in known)/len(known),
            'always_tie_baseline':{'accuracy':sum(r['semantic_label']=='Tie' for r in known)/len(known),
                'macro_f1':float(f1_score([LABELS.index(r['semantic_label']) for r in known],np.ones(len(known),int),labels=[0,1,2],average='macro',zero_division=0))},
            'training_teacher_audit':json.loads((OUT/'qwen27_training_audit.json').read_text()),
            'calibration':params,'models':{}}
    logits={}
    for arm,path in paths.items():
        model,cfg=C.load_model(path);model.to('cuda');start=time.monotonic()
        z=C.infer(model,enc);torch.cuda.synchronize();elapsed=time.monotonic()-start;logits[arm]=z
        np.save(OUT/f'{arm}_test_logits.npy',z);p=C.probabilities(z,params[arm]['temperature'])
        eligible=[i for i,r in enumerate(test) if C.allowed(r)]
        eligible.sort(key=lambda i:(-float(p[i,0]),test[i]['id']))
        top=np.zeros(len(test),bool);top[eligible[:int(np.ceil(.1*len(test)))]]=True
        stats=admission(test,p,params[arm]['threshold'])
        report['models'][arm]={'semantic':classification(test,z,params[arm]['temperature']),
           'bleu':classification(test,z,params[arm]['temperature'],'bleu_label'),
           'semantic_uncalibrated':classification(test,z,1.),
           'calibrated_acceptance':stats,'default_p_better_05':admission(test,p,{'p_better':.5,'p_worse_max':1.}),
           'top_10_percent':admission(test,p,mask=top),'inference_s':elapsed,
           'pilot_operating_gate_passed':params[arm]['threshold'] is not None and stats['coverage']>=.1 and
               (stats['confirmed_precision_lower'] or 0)>=.8 and stats['worst_case_utility_per_pair']>0}
        del model;torch.cuda.empty_cache()
    report['paired_bootstrap']=paired_bootstrap(test,logits,params)
    report['format_diagnostic']=format_diagnostic(test,logits,params)
    report['limitations']=['Semantic labels are from the same teacher model for train and evaluation, with order consensus, not human gold.',
        'Training uses compact winner-only output; heldout labeling requests short reasons. Both use 8-pair input batches. Output format can influence the teacher.',
        'Small changed-pair pilot, one training seed; uncertainty is retained and explicitly reported.',
        'New test sources were held out from both new models but originated in an older collection; old 20k checkpoint excluded.',
        'GLM9 teacher rejected by training-only audit before training/test inspection; original outputs preserved.',
        'Replacement teacher agrees with AI blind review on 24/32 sampled training pairs; semantic labels remain noisy and are not human gold.',
        'Passing a pilot operating point would still require independent semantic labels and real full_static/full_online rollouts.']
    dump(OUT/'comparison_report.json',report)
    write_report(report)
    status('complete',report=str(OUT/'COMPARISON_ZH.md'),semantic_pilot_gate_passed=report['models']['semantic']['pilot_operating_gate_passed'])
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def write_report(r):
    def fmt(v):return '—' if v is None else f'{v:.4f}'
    lines=['# BLEU 标签与语义标签：Laya multilingual 配对试验','',
       '这是小规模、教师相对的分类对照，不能视为人工翻译质量或 full_online 主实验。',
       '',f"同一原始 512 对候选，双方共同保留 {r['training']['matched_pairs']} 对；两臂均从 laya-multilingual 原始权重初始化，训练预算、输入及开发集选择规则相同。",
       f"训练语义标签：{r['training']['semantic_counts']}。相同输入上的 BLEU 标签：{r['training']['bleu_counts_on_same_rows']}。",
       f"测试 {r['test_pairs']} 对 / 同数独立原文；确定标签 {r['resolved_pairs']} 对，分布 {r['test_label_counts']}。",
       '', '| 模型 | 语义 Better AUC | 语义 Macro F1 | 语义准确率 | Better 召回 | 校准接受数 | 接受后确认改善率下界 |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for arm,m in r['models'].items():
        s=m['semantic'];a=m['calibrated_acceptance']
        lines.append(f"| {arm} | {fmt(s['auc_better_all'])} | {fmt(s['macro_f1'])} | {fmt(s['accuracy'])} | {fmt(s['better_recall'])} | {a['accepted']} | {fmt(a['confirmed_precision_lower'])} |")
    lines+=['',f"全判 Tie：准确率 {fmt(r['always_tie_baseline']['accuracy'])}，Macro F1 {fmt(r['always_tie_baseline']['macro_f1'])}。",
       f"训练 BLEU/语义标签一致率 {r['training']['label_agreement']:.1%}；测试确定标签一致率 {r['semantic_vs_bleu_agreement_on_resolved_test']:.1%}。",
       '',f"配对 bootstrap（semantic − BLEU）95% 区间：{r['paired_bootstrap']['ci95']}。",
       '', '接受阈值仅在独立校准集搜索：至少 10% 覆盖、至少 80% 确认改善率、正的最坏情况效用。接受到 Uncertain 时计入未确认改善；无可行阈值则停止接受，不能解释为高精度成功。',
       '',f"semantic 在留出测试上的试验门槛通过：{r['models']['semantic']['pilot_operating_gate_passed']}。",
       '', '最初 GLM-4-9B 教师虽通过简单预检，真实候选审核发现会偏好原文没有的实体信息，故在训练及测试查看前弃用并完整归档。替换后的训练与评估共用 Qwen 27B 教师；审核来自 Codex AI，不能称为人工金标。',
       '', '替换教师通过 32/32 个简单预检样例，但真实训练样本盲审仅 24/32 一致（审核为 AI，抽样含定向补充），仍有同义表达、数值格式和风格偏好分歧。不能据此声称训练标签已经可靠。',
       '', '局限：样本小、单次初始化；同教师训练评估存在共同偏差；两端教师输出格式不同；部分标签不确定。即使数值改善，也需要独立语义审查及真实固定库/在线库实验后才能判断方法有效。',
       '',f"模型保存：{OUT/'bleu/best'} 与 {OUT/'semantic/best'}。",
       '原始预测、混淆矩阵、置信区间、标签、校准锁和教师理由全部保存在本目录。']
    diag=r['format_diagnostic'];lines += ['', '预先声明的输出格式敏感性诊断（不重新选模型或校准阈值）：', '',
       f"测试集两种教师输出格式的全部标签一致率 {diag['folds']['test']['agreement_all']:.1%}；两者均确定时的一致率 {diag['folds']['test']['agreement_both_resolved']:.1%}。"]
    for arm,m in diag['models'].items():
        lines.append(f"- {arm}：改用精简输出教师标签评分时，Better AUC {fmt(m['classification']['auc_better_all'])}，Macro F1 {fmt(m['classification']['macro_f1'])}。")
    text='\n'.join(lines)+'\n';(OUT/'COMPARISON_ZH.md').write_text(text)
    (ROOT/'reports/LAYA_LABEL_COMPARISON.md').write_text(text)


if __name__=='__main__':run()
