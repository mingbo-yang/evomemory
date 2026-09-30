"""Source-disjoint pilot: absolute MSE versus MSE + pairwise ranking.
No production weights/configurations are modified. Test is evaluated only after
checkpoint selection and threshold calibration on validation.
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('BERT_STUDY_GPU', '1')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse, json, hashlib, random, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizerFast
from scipy.stats import spearmanr, rankdata

BASE = '/mnt/huawei/wwq/wwq/huggingface/model/bert-base-multilingual-cased'
CSV = '/mnt/huawei/wwq/model/aaa_experiment/control_n/merged_llm_data.csv'
OLD = '/mnt/huawei/wwq/model/aaa_experiment/control_n/model/layers_3_best.pth'
ROOT = Path(__file__).resolve().parent / 'runs/bert_feedback_study'

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))

def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'data.json'
    if cache.exists():
        return json.loads(cache.read_text())
    print('Reading historical CSV', flush=True)
    cols = ['source_text'] + [f'round_{r}_{suffix}' for r in range(6) for suffix in ('translation', 'bleu')]
    df = pd.read_csv(CSV, usecols=cols)
    sources = sorted(set(df.source_text.dropna().astype(str)))
    rng = np.random.default_rng(42)
    rng.shuffle(sources)
    n = len(sources)
    groups = {'train': sources[:int(.7*n)], 'validation': sources[int(.7*n):int(.85*n)], 'test': sources[int(.85*n):]}
    lookup = {s: fold for fold, ss in groups.items() for s in ss}
    pairs = {fold: {} for fold in groups}
    conflicts = 0
    labels = {}
    inconsistent = set()
    for row in df.to_dict('records'):
        source = row['source_text']
        if not isinstance(source, str):
            continue
        for r in range(1, 6):
            before, after = row[f'round_{r-1}_translation'], row[f'round_{r}_translation']
            y0, y1 = row[f'round_{r-1}_bleu'], row[f'round_{r}_bleu']
            if not isinstance(before, str) or not isinstance(after, str) or not before.strip() or not after.strip() or before == after:
                continue
            if not np.isfinite(y0) or not np.isfinite(y1):
                continue
            for answer, score in [(before,y0),(after,y1)]:
                label_key = (source,answer)
                if label_key in labels and abs(labels[label_key]-score)>1e-6:
                    inconsistent.add(label_key)
                labels[label_key] = score
            key = (source, before, after)
            rec = [source, before, after, float(y0), float(y1)]
            existing = pairs[lookup[source]].get(key)
            if existing and (abs(existing[3]-y0)>1e-6 or abs(existing[4]-y1)>1e-6):
                conflicts += 1
            pairs[lookup[source]][key] = rec
    print(f'Quarantining {len(inconsistent)} source-answer keys with inconsistent labels ({conflicts} duplicate pair conflicts)',flush=True)
    result, counts = {}, {}
    for fold, cap in [('train',6000),('validation',2000),('test',2000)]:
        raw_values = list(pairs[fold].values())
        values = [r for r in raw_values if (r[0],r[1]) not in inconsistent and (r[0],r[2]) not in inconsistent]
        random.Random(42).shuffle(values)
        counts[fold] = {'quarantined_pairs':len(raw_values)-len(values), 'available_pairs': len(values), 'used_pairs': min(cap,len(values)), 'source_groups':len(groups[fold])}
        result[fold] = values[:cap]
    assert not set(groups['train']) & set(groups['validation'])
    assert not set(groups['train']) & set(groups['test'])
    assert not set(groups['validation']) & set(groups['test'])
    dump(ROOT/'source_split.json', groups)
    dump(ROOT/'data_manifest.json', {'raw_rows':len(df), 'unique_sources':n, 'counts':counts, 'source_overlap':0, 'duplicate_label_conflicts':conflicts, 'quarantined_source_answer_keys':len(inconsistent), 'data_sha256':hashlib.sha256(json.dumps(result,ensure_ascii=False).encode()).hexdigest()})
    dump(cache, result)
    print('Data prepared: '+json.dumps(counts), flush=True)
    return result

class Model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(BASE,local_files_only=True) if pretrained else BertModel(BertConfig.from_pretrained(BASE,local_files_only=True))
        layers = [nn.Linear(768,512), nn.ReLU(), nn.Dropout(.3)]
        for _ in range(3): layers.extend([nn.Linear(512,512),nn.ReLU(),nn.Dropout(.3)])
        layers.append(nn.Linear(512,1))
        self.classifier = nn.Sequential(*layers)
    def forward(self, **batch):
        return self.classifier(self.bert(**batch).pooler_output).squeeze(-1)

def tensor_batch(records, tokenizer):
    # Interleaved before/after; identical input budget for both training arms.
    sources = [r[0] for r in records for _ in range(2)]
    candidates = [v for r in records for v in r[1:3]]
    enc = tokenizer(sources,candidates,padding=True,truncation=True,max_length=512,return_tensors='pt')
    return {k:v.cuda() for k,v in enc.items()}, torch.tensor([r[3:5] for r in records],device='cuda',dtype=torch.float32)

def summarize(pred, data, threshold=.01):
    truth = np.asarray([r[3:5] for r in data])
    dg = pred[:,1]-pred[:,0]; dy = truth[:,1]-truth[:,0]
    positive = dy > 1e-6; negative = dy < -1e-6; mask = positive|negative
    npos, nneg = int(positive[mask].sum()), int(negative[mask].sum())
    auc = float((rankdata(dg[mask])[positive[mask]].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else None
    accept = np.zeros(len(data),dtype=bool) if threshold is None else dg > threshold
    num = int(accept.sum())
    rho = float(spearmanr(dg,dy).statistic)
    return {'n':len(data),'absolute_mse':float(np.mean((pred-truth)**2)), 'gain_mse':float(np.mean((dg-dy)**2)), 'gain_spearman':rho if math.isfinite(rho) else None, 'improvement_auc':auc, 'positive_fraction':float(positive.mean()), 'threshold':threshold, 'accepted':num,'coverage':float(accept.mean()),'accept_precision':float(positive[accept].mean()) if num else None,'accepted_improve':int((accept&positive).sum()),'accepted_degrade':int((accept&negative).sum()),'accepted_tie':int((accept&~mask).sum()),'mean_gain_per_pair':float(np.where(accept,dy,0).mean()),'mean_gain_when_accepted':float(dy[accept].mean()) if num else None}

@torch.inference_mode()
def predict(model, data, tokenizer):
    model.eval(); output=[]
    for i in range(0,len(data),16):
        enc,_ = tensor_batch(data[i:i+16],tokenizer)
        # Float32 inference matches deployed quality evaluator.
        output.append(model(**enc).reshape(-1,2).float().cpu().numpy())
    return np.concatenate(output)

def calibrate(pred, data):
    table = [summarize(pred,data,t) for t in [.0,.01,.02,.03,.05,.075,.1,.15,.2]]
    candidates = [m for m in table if m['accepted']>=30 and m['accept_precision']>=.8]
    best = max(candidates,key=lambda m:m['coverage']) if candidates else None
    return best['threshold'] if best else None, table

def evaluate(model, data, tokenizer, name):
    val = predict(model,data['validation'],tokenizer)
    threshold, table = calibrate(val,data['validation'])
    test = predict(model,data['test'],tokenizer)
    np.savez(ROOT/f'{name}_predictions.npz',validation=val,test=test)
    report = {'validation_threshold_search':table,'calibrated_threshold':threshold,'calibration_passed':threshold is not None,'test_fixed':summarize(test,data['test']), 'test_calibrated':summarize(test,data['test'],threshold)}
    dump(ROOT/f'{name}_evaluation.json',report)
    print(name+' FINAL '+json.dumps(report['test_fixed']),flush=True)
    return report

def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = prepare()
    dump(ROOT/'protocol.json',{'scope':'en-to-zh bounded diagnostic pilot, not a main experiment','physical_gpu':1,'epochs':args.epochs,'pair_batch_size':16,'max_length':512,'seed':42,'split':'70/15/15 source groups; caps train6000 val2000 test2000 changed adjacent-round pairs','initialization':'same pretrained multilingual BERT and random regression head, not legacy finetuned weights','optimizer':'AdamW encoder2e-5 head1e-3 weight_decay.01, linear schedule','arms':{'absolute':'MSE of both absolute scores','pairwise':'same MSE + 0.1 softplus(-sign(delta_label)*delta_prediction/0.1), ties delta_prediction squared'},'checkpoint_selection':'minimum validation gain MSE, same for both new arms','threshold_calibration':'validation precision>=0.8 and accepted>=30, maximize coverage on fixed grid; disable acceptance if none qualifies','legacy_warning':'legacy checkpoint saw original sources; its test metrics are contaminated diagnostic only','label':'existing char BLEU in [0,1]; reference only supplies supervised labels, not model inputs'})
    tokenizer = BertTokenizerFast.from_pretrained(BASE,local_files_only=True)
    for arm in ['absolute','pairwise']:
        if (ROOT/f'{arm}_evaluation.json').exists():
            print('Completed arm exists, skipping '+arm,flush=True); continue
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        model = Model().cuda()
        optimizer = torch.optim.AdamW([{'params':model.bert.parameters(),'lr':2e-5},{'params':model.classifier.parameters(),'lr':1e-3}],weight_decay=.01)
        total_steps = math.ceil(len(data['train'])/16)*args.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0,1-step/total_steps))
        best=float('inf'); history=[]
        for epoch in range(args.epochs):
            order = np.random.default_rng(42+epoch).permutation(len(data['train']))
            model.train(); loss_sum=0.; start=time.time()
            for step,i in enumerate(range(0,len(order),16)):
                records=[data['train'][j] for j in order[i:i+16]]
                enc,target=tensor_batch(records,tokenizer)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(**enc).reshape(-1,2).float()
                    loss=nn.functional.mse_loss(pred,target)
                    if arm=='pairwise':
                        dy=target[:,1]-target[:,0]; dg=pred[:,1]-pred[:,0]
                        rankloss=torch.where(dy.abs()>1e-6,nn.functional.softplus(-dy.sign()*dg/.1),dg.square())
                        loss=loss+.1*rankloss.mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step(); scheduler.step(); loss_sum+=loss.item()*len(records)
                if step%100==0: print(f'{arm} epoch={epoch+1} step={step} loss={loss.item():.5f}',flush=True)
            val=predict(model,data['validation'],tokenizer)
            metrics=summarize(val,data['validation'])
            entry={'epoch':epoch+1,'train_objective':loss_sum/len(order),'elapsed_s':time.time()-start,'validation':metrics}
            history.append(entry); dump(ROOT/f'{arm}_learning_curve.json',history)
            print(arm+' EPOCH '+json.dumps(entry),flush=True)
            if metrics['gain_mse']<best:
                best=metrics['gain_mse']; torch.save(model.cpu().state_dict(),ROOT/f'{arm}_best.pth'); model.cuda()
                dump(ROOT/f'{arm}_selection.json',{'epoch':epoch+1,'validation_gain_mse':best})
        del optimizer,scheduler
        model.load_state_dict(torch.load(ROOT/f'{arm}_best.pth',map_location='cpu',weights_only=True))
        evaluate(model,data,tokenizer,arm)
        del model; torch.cuda.empty_cache()
    if not (ROOT/'legacy_evaluation.json').exists():
        model=Model(False)
        model.load_state_dict(torch.load(OLD,map_location='cpu',weights_only=True),strict=True)
        model.cuda(); evaluate(model,data,tokenizer,'legacy')
    print('STUDY COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=5)
    run(parser.parse_args())
