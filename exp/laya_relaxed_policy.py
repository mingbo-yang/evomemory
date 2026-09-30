"""User-authorized exploratory operating point, selected without new test feedback."""
from collections import Counter
import json
import numpy as np
import laya_acceptance_common as C
from semantic_label_common import OUT as MODEL_OUT,read_jsonl


def choose_policy():
    rows=read_jsonl(MODEL_OUT/'inputs/threshold_calibration.jsonl')
    labels={r['id']:r['label'] for r in read_jsonl(MODEL_OUT/'labels/threshold_calibration.jsonl')}
    lock=json.loads((MODEL_OUT/'comparison_evaluation_lock.json').read_text())
    temp=lock['models']['semantic']['temperature']
    p=C.probabilities(np.load(MODEL_OUT/'semantic_threshold_logits.npy'),temp)
    grid=[];feasible=[]
    for tb in [round(.05+.025*i,3) for i in range(23)]:
        for tw in [1.,.8,.6,.4,.3,.2]:
            mask=C.accepted_mask(rows,p,{'p_better':tb,'p_worse_max':tw})
            counts=Counter(labels[r['id']] for r,m in zip(rows,mask) if m);n=int(mask.sum())
            b,w,u=counts['Better'],counts['Worse'],counts['Uncertain']
            s={'p_better':tb,'p_worse_max':tw,'accepted':n,'counts':dict(counts),
               'coverage':n/len(rows),'confirmed_precision':b/n if n else None,
               'net_confirmed':b-w,'net_worst_case':b-w-u}
            grid.append(s)
            if n>=5 and b/n>=.5 and b-w-u>0:feasible.append(s)
    assert feasible,'No approximately-50% operating point on the existing calibration set'
    best=max(feasible,key=lambda r:(r['net_worst_case'],r['accepted'],r['confirmed_precision'],
                                   -r['p_worse_max'],-r['p_better']))
    return {'version':'semantic-relaxed-v1','checkpoint':str(MODEL_OUT/'semantic/best'),
            'temperature':temp,'p_better':best['p_better'],'p_worse_max':best['p_worse_max'],
            'selection':'existing threshold fold only: >=5 accepted, >=50% confirmed improvement counting Uncertain as unconfirmed, positive worst-case utility; max worst-case net, then coverage, precision, smaller pW cap, smaller pB cut',
            'calibration':best,'grid':grid,'new_test_feedback_used':False,
            'interpretation':'exploratory starting point, not a guarantee of 50% precision after best-of-4 trajectory selection'}


def select_candidate(candidates,policy,mode='laya'):
    valid=[(i,c) for i,c in enumerate(candidates) if c.get('valid') and c.get('changed')]
    if mode=='unfiltered':return valid[0][0] if valid else None
    eligible=[(i,c) for i,c in valid if c.get('probabilities') is not None
              and c['probabilities'][0]>=policy['p_better'] and c['probabilities'][2]<=policy['p_worse_max']]
    return max(eligible,key=lambda t:(t[1]['probabilities'][0],-t[0]))[0] if eligible else None
