"""Prepare source-aware, score-blind review items from the frozen rollout."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/laya_relaxed_flow_v1'
OUT = RUN / 'comet_posthoc_v1'
ARMS = ('laya_static', 'laya_bleu_static')


def key(source, current, candidate):
    return hashlib.sha256(json.dumps([source, current, candidate], ensure_ascii=False).encode()).hexdigest()


def main():
    accepted, rejected = [], []
    for arm in ARMS:
        for item in json.loads((RUN / arm / 'accepted_feedback.json').read_text()):
            accepted.append({'arm': arm, 'sample_id': item['sample_id'],
                             'round': item['round'], 'source': item['source'],
                             'current': item['before'], 'candidate': item['after'],
                             'kind': 'accepted'})
        for trace in json.loads((RUN / arm / 'results.json').read_text()):
            for step in trace['rounds']:
                for candidate in step['candidates']:
                    if not candidate['valid'] or not candidate['changed']:
                        continue
                    if step['accepted'] and candidate['slot'] == step['selected_slot']:
                        continue
                    rejected.append({'arm': arm, 'sample_id': trace['sample_id'],
                                     'round': step['round'], 'source': trace['source'],
                                     'current': step['before'], 'candidate': candidate['text'],
                                     'kind': 'unaccepted'})
    assert len(accepted) == 38
    chosen = list(accepted)
    used = {key(x['source'], x['current'], x['candidate']) for x in chosen}
    rng = random.Random(20260930)
    rng.shuffle(rejected)
    rejected_by_arm = Counter()
    for item in rejected:
        if rejected_by_arm[item['arm']] >= 20:
            continue
        k = key(item['source'], item['current'], item['candidate'])
        if k in used:
            continue
        chosen.append(item)
        used.add(k)
        rejected_by_arm[item['arm']] += 1
    assert rejected_by_arm == Counter({'laya_static':20, 'laya_bleu_static':20})
    rng.shuffle(chosen)
    mapping = {}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'BLIND_REVIEW.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'English source', 'Translation A',
                                'Translation B', 'Winner (A/B/Tie/Uncertain)',
                                'Error type', 'Reason'])
        writer.writeheader()
        for i, item in enumerate(chosen):
            ident = f'R{i+1:03d}'
            current_is_a = rng.random() < .5
            mapping[ident] = {k:v for k,v in item.items() if k not in ('source','current','candidate')}
            mapping[ident]['current_is_A'] = current_is_a
            mapping[ident]['triple_sha256'] = key(item['source'], item['current'], item['candidate'])
            writer.writerow({'id': ident, 'English source': item['source'],
                             'Translation A': item['current'] if current_is_a else item['candidate'],
                             'Translation B': item['candidate'] if current_is_a else item['current'],
                             'Winner (A/B/Tie/Uncertain)': '', 'Error type': '', 'Reason': ''})
    (OUT / 'review_mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2)+'\n')
    (OUT / 'REVIEW_INSTRUCTIONS_ZH.md').write_text(
        '# 译文修改盲评说明\n\n'
        '共 78 行：两版 Laya 实际采纳的 38 次修改，加上各 20 次未采纳候选。'
        'A/B 顺序随机；表中不提供方法、采纳情况、参考译文或模型分数。'
        '同一原文在不同状态或方法中可能出现。\n\n'
        '请只看英文原文和 A/B，选 A、B、Tie 或 Uncertain。'
        '质量同时考虑忠实性、遗漏/添加、流畅性；纯标点或等质同义改写可选 Tie。'
        '记录关键错误类型和一句理由。最好由两位互不知晓对方判断的人独立填写；'
        '发生分歧时，再核对原文并仲裁。\n\n'
        'review_mapping.json 用于评完后恢复实际修改方向和方法；评审前不要打开。'
        '本文件尚无人工标注，不应计算人工精度。\n')
    print(json.dumps({'review_rows':len(chosen),'accepted_decisions':len(accepted),
                      'unaccepted_by_arm':dict(rejected_by_arm)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
