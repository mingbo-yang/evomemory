import json
from pathlib import Path
from collections import Counter
root=Path('/mnt/huawei/ymb/aaai2027/exp'); data=root/'runs/acceptance_data_v2'
review=json.loads((data/'label_preparation/assistant_review.json').read_text())
samples=json.loads((data/'train_margin_review.json').read_text())
rs=review['reviews']
rows=[json.loads(s) for s in (data/'pairs/train.jsonl').read_text().splitlines()]
q={'wmt19_en_zh/acceptance_v1/train/018316','wmt19_en_zh/acceptance_v1/train/032431','wmt19_en_zh/acceptance_v1/train/035677'}
rows=[r for r in rows if r['valid'] and r['sample_id'] not in q]
checked=[r for r in rs if r['assistant_label']!='Uncertain' and r['sample_id'] not in q]
lines=['# Acceptance data: label preparation review','',
'All 80 presampled training pairs were reviewed by the Codex assistant. This is AI-assisted qualitative review, not human annotation, not blinded, and not an independent gold evaluation. Twenty pairs came from each absolute-delta bin; the sample is not population-proportional. Do not extrapolate its agreement counts into whole-dataset label precision.',
'', '## Findings','',
'Assistant judgments: '+str(dict(Counter(r['assistant_label'] for r in rs)))+'.',
'', 'Simple lexical edits can move the existing sentence feedback by more than 0.05 and even 0.68. Enlarging epsilon does not establish reliable semantic labels. Shared translation errors do not imply a relative improvement/degradation; several pairs preserve the same incorrect translation.',
'', 'Three training sources have a clearly misaligned or incomplete reference. Their source/reference fields were checked against the original parquet and match exactly; no collector index error was found. All 11 associated pairs were quarantined in the derived dataset. Seven other invalid pairs were filtered. Originals are preserved. This is not an exhaustive reference-quality audit of all sources.',
'', '## Candidate-margin diagnostics','',
'Counts below are descriptive and do not freeze a label rule. Review agreement excludes four Uncertain judgments and three misaligned-reference cases, leaving 73 pairs.',
'', '| ε | Train Better | Train Tie | Train Worse | Review matching labels / 73 | Review matching directional labels / predicted directional |',
'|---:|---:|---:|---:|---:|---:|']
stats=[]
for e in [0,.005,.01,.02,.03,.05]:
 def lab(d):return 'Better' if d>e else 'Worse' if d< -e else 'Tie'
 cc=Counter(lab(r['delta']) for r in rows)
 match=sum(lab(r['delta'])==r['assistant_label'] for r in checked)
 d=[r for r in checked if lab(r['delta'])!='Tie']
 dm=sum(lab(r['delta'])==r['assistant_label'] for r in d)
 lines.append(f"| {e} | {cc['Better']} | {cc['Tie']} | {cc['Worse']} | {match}/73 | {dm}/{len(d)} |")
 stats.append({'epsilon':e,'training_counts':dict(cc),'review_total':73,'review_matches':match,'directional_review_count':len(d),'directional_review_matches':dm})
lines += ['', 'The greater overall match for a larger margin mainly reflects converting cases into Tie. It should not be mistaken for evidence of high-quality Better/Worse supervision. A weak-feedback export can still support a proxy-objective experiment if explicitly chosen; a semantic verifier needs a better justified label source.',
'', '## Per-pair notes','']
for note,s in zip(rs,samples):
 lines += [f"### {note['review_index']}: {note['sample_id']}",'',
           f"- AI-assisted label: {note['assistant_label']}; delta={note['delta']:.8f}; reference issue={note['reference_issue']}",
           '- Source: '+s['source'],'- Before: '+s['before'],'- Candidate: '+s['candidate'],
           '- Reference (review only): '+s['reference_for_human_review_only'],
           '- Reason: '+note['assistant_reason'],'']
(root/'reports/ACCEPTANCE_LABEL_REVIEW.md').write_text('\n'.join(lines))
(data/'label_preparation/margin_diagnostics.json').write_text(json.dumps({'review_limitations':review['blinding'],'candidate_margins':stats,'frozen_epsilon':None},ensure_ascii=False,indent=2))
print('Review report saved; epsilon remains unset.')
