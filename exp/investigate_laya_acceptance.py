"""Read-only diagnosis: official XNLI control and train/development probes.

No retraining, checkpoint selection, threshold tuning, or new main-test probes.
"""
from datetime import datetime, timezone
import json
import time
import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from laya_acceptance_common import *
import laya

DEST = OUT/'investigation'
NLI_LABELS = ['entailment', 'neutral', 'contradiction']
NLI_QUESTION = {'relation': {'type': 'choice',
    'instructions': 'What is the relationship between `premise` and `hypothesis`?',
    'criteria': {'entailment': 'the premise implies the hypothesis is true',
                 'neutral': 'the premise neither implies nor contradicts the hypothesis',
                 'contradiction': 'the premise implies the hypothesis is false'}}}
STANDARD = {'acceptance': {'type': 'choice', 'instructions': QUESTION['ins'], 'criteria': CRITERIA}}
RICH = {'acceptance': {'type': 'choice',
    'instructions': 'Compare the Current Answer and Candidate Answer as Chinese translations of the Source. Judge preservation of source meaning, factual accuracy, completeness, and Chinese grammar. Mere synonymous wording changes are Tie.',
    'criteria': {
        'Better': 'Candidate fixes a mistranslation, missing or added meaning, factual error, or grammatical error without introducing a comparable new error.',
        'Tie': 'Both answers have equivalent meaning and quality; changes only rephrase synonyms, punctuation, or style.',
        'Worse': 'Candidate introduces a mistranslation, missing or added meaning, factual error, or grammatical error without a comparable improvement.'}}}


def api_probs(agent, states, questions, key='acceptance', labels=LABELS):
    result = agent.predict_batch(states, questions, batch_size=32)
    return np.array([[r['answers'][key]['probabilities'][label] for label in labels] for r in result])


def synthetic_pairs():
    # Hand-constructed sanity checks, not representative test data or human annotation.
    items = [
        ('The meeting starts at nine in the morning.', '会议在上午九点开始。', '会议在晚上九点开始。', '会议上午九点开始。'),
        ('Alice bought three red apples.', '爱丽丝买了三个红苹果。', '爱丽丝买了五个红苹果。', '爱丽丝购买了三个红苹果。'),
        ('The medicine must not be taken with alcohol.', '这种药物不得与酒精一起服用。', '这种药物必须与酒精一起服用。', '服用这种药物时不能饮酒。'),
        ('The train leaves on Monday, not Tuesday.', '火车星期一出发，不是星期二。', '火车星期二出发，不是星期一。', '列车周一出发，而不是周二。'),
        ('The company lost two million dollars last year.', '公司去年亏损了两百万美元。', '公司去年盈利了两百万美元。', '这家公司去年损失了两百万美元。'),
        ('Tom is older than his sister.', '汤姆比他的妹妹年长。', '汤姆比他的姐姐年轻。', '汤姆的年龄比他的妹妹大。'),
        ('Please close the window before you leave.', '请在离开前关上窗户。', '请在离开后打开窗户。', '离开之前，请把窗户关好。'),
        ('Only employees can enter this room.', '只有员工可以进入这个房间。', '所有人都可以进入这个房间。', '这个房间仅限员工进入。'),
    ]
    rows = []
    for i, (source, good, bad, paraphrase) in enumerate(items):
        for kind, a, b, label in [('fix', bad, good, 'Better'), ('damage', good, bad, 'Worse'),
                                   ('identical', good, good, 'Tie'), ('paraphrase', good, paraphrase, 'Tie')]:
            rows.append({'id': f'sanity-{i}-{kind}', 'kind': kind,
                         'input': {'source': source, 'current': a, 'candidate': b}, 'label': label})
    return rows


def run():
    torch.set_num_threads(4)
    DEST.mkdir(exist_ok=True)
    lock_hash = digest(OUT/'evaluation_lock.json')
    protocol = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': digest(Path(__file__)), 'scope': 'diagnosis only, original experiment stays frozen',
        'xnli': 'same first 300 Chinese test rows and same official prompt; base and fine-tuned checkpoints',
        'training_fit': 'all 20002 natural training pairs; no swapped evaluation',
        'development_probes': ['original text prompt', 'JSON input with original instructions', 'explicit semantic rubric', 'swap current/candidate'],
        'synthetic': '32 hand-constructed easy cases; qualitative sanity check, not population estimate',
        'main_test': 'no new inference or tuning on main test'}
    dump(DEST/'protocol.json', protocol)
    nli = pq.read_table(DEST/'xnli_zh_test.parquet').slice(0, 300).to_pylist()
    nli_states = [{'premise': r['premise'], 'hypothesis': r['hypothesis']} for r in nli]
    nli_y = np.array([r['label'] for r in nli])
    train, td, _ = load_fold('train'); dev, dd, _ = load_fold('development')
    audit = read_jsonl(DATA/'audit/train.jsonl')
    by_id = {r['id']: r for r in audit}
    sanity = synthetic_pairs(); dump(DEST/'synthetic_cases.json', sanity)
    tok = get_tokenizer(); tr_encoded = EncodedPairs(train, tok); de_encoded = EncodedPairs(dev, tok)
    flipped = [{'id': r['id'], 'input': {'source': r['input']['source'], 'current': r['input']['candidate'],
               'candidate': r['input']['current']}, 'label': LABELS[2-LABELS.index(r['label'])]} for r in dev]
    flipped_encoded = EncodedPairs(flipped, tok)
    result = {'protocol': protocol, 'models': {}}
    official = json.loads((VENDOR/'research/results/t4_colab_benchmark.json').read_text())
    result['official_xnli_zh'] = official['suites']['xnli.zh']['laya-multilingual']['raw']
    for name, path in [('base', BASE), ('finetuned', OUT/'best')]:
        start = time.time(); print('MODEL', name, flush=True)
        agent = laya.Agent(str(path), device='cuda', compile=False, fast=False)
        pn = api_probs(agent, nli_states, NLI_QUESTION, 'relation', NLI_LABELS)
        item = {'xnli_zh': {'n': len(nli), 'accuracy': float(accuracy_score(nli_y, pn.argmax(1))),
                          'confusion': confusion_matrix(nli_y, pn.argmax(1), labels=[0,1,2]).tolist()}}
        print('XNLI', name, item['xnli_zh'], flush=True)
        np.save(DEST/f'{name}_xnli_probabilities.npy', pn)
        ztr = infer(agent.model, tr_encoded); np.save(DEST/f'{name}_train_logits.npy', ztr)
        item['train'] = metrics(train, ztr, td)
        item['train_by_temperature'] = {}
        for temperature in [.1, .7]:
            idx = [i for i, r in enumerate(train) if by_id[r['id']]['temperature'] == temperature]
            item['train_by_temperature'][str(temperature)] = metrics([train[i] for i in idx], ztr[idx], [td[i] for i in idx])
        zd = infer(agent.model, de_encoded)
        item['development'] = metrics(dev, zd, dd)
        for mode, states, question in [
            ('json_state', [r['input'] for r in dev], STANDARD),
            ('semantic_rubric', [state_text(r['input']) for r in dev], RICH),
        ]:
            p = api_probs(agent, states, question)
            item[mode] = metrics(dev, np.log(np.maximum(p, 1e-12)), dd)
            np.save(DEST/f'{name}_dev_{mode}_probabilities.npy', p)
        zf = infer(agent.model, flipped_encoded)
        p = probabilities(zd); reversed_p = probabilities(zf)[:, ::-1]
        changed = np.array([r['input']['current'] != r['input']['candidate'] for r in dev])
        item['swap_consistency'] = {'changed_argmax_agreement': float((p[changed].argmax(1)==reversed_p[changed].argmax(1)).mean()),
                                    'changed_mean_abs_probability_difference': float(abs(p[changed]-reversed_p[changed]).mean())}
        ps = api_probs(agent, [state_text(r['input']) for r in sanity], STANDARD)
        sy = np.array([LABELS.index(r['label']) for r in sanity]); pred = ps.argmax(1)
        item['synthetic'] = {'n': len(sanity), 'accuracy': float((pred==sy).mean()),
            'by_kind': {k: float(np.mean([pred[i]==sy[i] for i, r in enumerate(sanity) if r['kind']==k])) for k in ['fix','damage','identical','paraphrase']},
            'rows': [{'id': r['id'], 'expected': r['label'], 'predicted': LABELS[int(pred[i])], 'probabilities': ps[i].tolist()} for i,r in enumerate(sanity)]}
        item['elapsed_s'] = time.time()-start
        result['models'][name] = item
        dump(DEST/'results.json', result)
        print('COMPLETED', name, 'train_auc',item['train']['auc_better_changed'],'dev_auc',item['development']['auc_better_changed'],flush=True)
        del agent; torch.cuda.empty_cache()
    assert digest(OUT/'evaluation_lock.json') == lock_hash
    result['primary_evaluation_unchanged'] = True
    dump(DEST/'results.json', result)
    print('INVESTIGATION_COMPLETE', flush=True)


if __name__ == '__main__':
    run()
