"""Training-only descriptive diagnostics; does not select epsilon or train models."""
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import acceptance_data as data
from core.scoring import Scorer

ROOT = Path(__file__).resolve().parent

def label_counts(rows, epsilon):
    counts = Counter('Better' if r['delta'] > epsilon else 'Worse' if r['delta'] < -epsilon else 'Tie' for r in rows)
    return {k: counts[k] for k in ('Better','Tie','Worse')}

def run():
    data.prepare()  # verifies frozen input hashes; never changes them
    records = list(data.iter_raw('train'))
    references = {r.sample_id:r.reference for r in data.read_manifest(data.OUT/'train_manifest.jsonl')}
    scorer = Scorer(data.TASK)
    @lru_cache(maxsize=None)
    def score(sample_id, text):
        return scorer.primary(references[sample_id], text)/100
    rows = [r for r in data.unique_rows(records) if r['valid']]
    for r in rows:
        r['delta'] = score(r['sample_id'],r['candidate'])-score(r['sample_id'],r['before'])
    changed = [r for r in rows if not r['identical']]
    margins = [0,.005,.01,.02,.03,.05]
    temperatures = {}
    for temp in [.1,.7]:
        occurrences = [r for rec in records for r in data.pair_rows(rec) if r['temperature']==temp]
        valid = [r for r in occurrences if r['valid']]
        keys = {(r['source'],r['before'],r['candidate']) for r in valid}
        temperatures[str(temp)] = {'raw_pairs':len(occurrences), 'valid_raw_pairs':len(valid),
                                  'valid_changed_raw_pairs':sum(not r['identical'] for r in valid),
                                  'unique_valid_pairs_within_temperature':len(keys)}
    transitions = Counter()
    for r in records:
        if len(r['states']) == 2:
            gain = score(r['sample_id'],r['states'][1]['before'])-score(r['sample_id'],r['states'][0]['before'])
            transitions['positive' if gain>1e-8 else 'negative' if gain< -1e-8 else 'zero'] += 1
    identity = Counter(r['sample_id'] for r in rows if r['identical'])
    result = {'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'scope':'train-only; interim snapshot; not final labels',
              'sources':len(records),'unique_valid_pairs':len(rows),'unique_valid_changed_pairs':len(changed),
              'unique_identical_pairs':len(rows)-len(changed),'sources_with_identity_pair':len(identity),
              'margin_sensitivity':[{'epsilon_0_to_1':e,'metric_points_0_to_100':100*e,
                                     'all_valid':label_counts(rows,e),'changed_only':label_counts(changed,e)} for e in margins],
              'by_temperature':temperatures, 'selected_second_state_feedback_sign':dict(transitions),
              'epsilon_selected':None,
              'notes':['Both temperature groups share sources and states; their counts cannot be added after per-temperature deduplication.',
                       'This is reference-derived sentence feedback, not verified semantic labels.',
                       'Temperature and threshold calibration folds and final test are not read here.']}
    data.dump(ROOT/'reports/acceptance_data_v2_interim.json', result)
    lines = ['# Acceptance data v2: interim training-only diagnostics','',f"Snapshot: {result['timestamp_utc']}",
             '',f"{len(records)} source trajectories; {len(rows)} unique valid pairs; {len(changed)} changed pairs. Collection is still running.",
             '', 'These are descriptive label counts, not a selection of epsilon. Feedback is the existing sentence proxy divided by 100.',
             '', '| ε (0–1) | Metric points | Better | Tie | Worse | Better (changed only) | Tie (changed only) | Worse (changed only) |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['margin_sensitivity']:
        a,b=row['all_valid'],row['changed_only']
        lines.append(f"| {row['epsilon_0_to_1']} | {row['metric_points_0_to_100']} | {a['Better']} | {a['Tie']} | {a['Worse']} | {b['Better']} | {b['Tie']} | {b['Worse']} |")
    lines += ['', 'Randomly chosen second-state feedback signs (computed afterward): '+json.dumps(dict(transitions)),
              '', 'The raw trajectories retain all candidates and flags. Filter `valid == true` for acceptance training; do not use reference feedback as verifier input. Keep exact identities visible when reporting distributions.',
              '', 'Manual review is still needed to distinguish lexical score movement from semantic improvement; increasing epsilon alone does not guarantee trustworthy labels.']
    (ROOT/'reports/ACCEPTANCE_DATA_V2_INTERIM.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    run()
