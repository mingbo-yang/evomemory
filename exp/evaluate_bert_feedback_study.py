"""Evaluate frozen checkpoints on existing full_online candidates (no new rollout)."""
import json, glob
import numpy as np
import torch
from transformers import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from bert_feedback_study import ROOT, BASE, OLD, Model, predict, summarize, dump
from core.quality_feedback import QualityPolicy


def main():
    torch.set_num_threads(4)
    exp=ROOT.parent.parent
    manifest={r['sample_id']:r for r in map(json.loads,(exp/'data/manifests/wmt19_en_zh__dev.jsonl').read_text().splitlines())}
    path=glob.glob(str(exp/'runs/optimized_flow_validation/full_online/seed42/wmt19_en_zh/qwen3-8b/dev/*/full_online.jsonl'))
    assert len(path)==1
    records=[]; official=[]; sources=set(); initial=[]
    def bleu(answer,ref):
        return sentence_bleu([list(ref)],list(answer),smoothing_function=SmoothingFunction().method1)
    for row in map(json.loads,open(path[0])):
        item=manifest[row['sample_id']]; source=item['source']; ref=item['reference']; current=row['initial_draft']
        initial.append([source,current,current,bleu(current,ref),bleu(current,ref)])
        sources.add(source)
        for rd in row['rounds']:
            candidate=rd.get('candidate')
            if candidate and candidate!=current:
                records.append([source,current,candidate,bleu(current,ref),bleu(candidate,ref)])
                official.append(rd['delta_offline'])
            if rd.get('accepted'): current=candidate
    groups=json.loads((ROOT/'source_split.json').read_text())
    assert not sources & set(sum(groups.values(),[]))
    tokenizer=BertTokenizer.from_pretrained(BASE,local_files_only=True)
    reports={}
    for arm in ['absolute','pairwise','legacy']:
        ckpt=OLD if arm=='legacy' else ROOT/f'{arm}_best.pth'
        model=Model(False); model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True));model.cuda()
        pred=predict(model,records,tokenizer)
        initpred=predict(model,initial,tokenizer)[:,0]
        threshold=json.loads((ROOT/f'{arm}_evaluation.json').read_text())['calibrated_threshold']
        policy=QualityPolicy(stop_threshold=.6)
        accepted=np.asarray([policy.accept(r[1],r[2],p[0],p[1])[0] for r,p in zip(records,pred)])
        delta=np.asarray(official)
        reports[arm]={'changed_round_candidates':len(records),'unique_source_before_after':len(set(tuple(r[:3]) for r in records)), 'source_overlap_with_historical_data':0,'fixed_threshold':summarize(pred,records), 'calibrated':summarize(pred,records,threshold),'fixed_threshold_with_length_gate':{'accepted':int(accepted.sum()),'improve':int((accepted&(delta>1e-6)).sum()),'degrade':int((accepted&(delta< -1e-6)).sum()),'tie':int((accepted&(np.abs(delta)<=1e-6)).sum()),'summed_sentence_metric_gain':float(delta[accepted].sum())},'initial_stop_at_0_6':int((initpred>.6).sum())}
        np.savez(ROOT/f'{arm}_external_predictions.npz',predictions=pred,initial=initpred)
        del model;torch.cuda.empty_cache()
    dump(ROOT/'external_candidates.json',records)
    dump(ROOT/'external_evaluation.json',{'warning':'Fixed candidates generated under legacy policy, repeated rounds are correlated; this is NOT a counterfactual full_online rollout or final corpus BLEU. External set never used for selection/calibration.','sample_count':len(sources),'results':reports})
    print(json.dumps(reports,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
