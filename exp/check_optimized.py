#!/usr/bin/env python
"""CPU preflight for local optimized assets, including legacy scoring parity."""
import argparse
import ast
import json
from pathlib import Path
import time
import torch

import core
from core.optimization_config import resolve_optimization
from core.quality_feedback import BertQualityEvaluator
from core.hybrid_retrieval import LocalSentenceEncoder, HybridExperienceRetriever
from core.bm25_fields import Experience


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).parent/'configs/optimized_feedback_v1.json'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    old_root = Path(__file__).resolve().parents[2]/'icml/exp/generation'
    report = {'kind': 'CPU asset and scoring smoke test; not an efficacy experiment', 'tasks': {}}
    examples = {
        'wmt19_en_zh': ('en_zh', 'predict_bleu_batch', [('The meeting starts at nine.', '会议九点开始。'), ('The meeting starts at nine.', '苹果很好吃。')]),
        'wmt19_zh_en': ('zh_en', 'predict_bleu_batch', [('会议九点开始。', 'The meeting starts at nine.'), ('会议九点开始。', 'Apples taste good.')]),
        'coedit_gec': ('gec', 'predict_gleu_batch', [('He go to school.', 'He goes to school.'), ('He go to school.', 'Blue green table.')]),
        'gigaword': ('gigaword_tiny', 'predict_rouge1_batch', [('The company reported rising profits on Tuesday.', 'company reports rising profits'), ('The company reported rising profits on Tuesday.', 'football match cancelled')]),
    }
    last = None
    for task, (folder, func, pairs) in examples.items():
        t0 = time.perf_counter()
        settings = resolve_optimization(args.config, task)
        evaluator = BertQualityEvaluator(settings['checkpoint'], settings['bert_base'], 'cpu')
        scores = evaluator.score_pairs(pairs)
        # Execute ONLY the old pure scoring function, never old model-launch code.
        source = old_root/folder/'llm_re_cosine_llama8b.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func)
        ns = {'torch': torch, 'MAX_BERT_SEQ_LEN': 512}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), ns)
        legacy = ns[func](evaluator.model, evaluator.tokenizer,
                          [a for a,b in pairs], [b for a,b in pairs], 'cpu')
        error = max(abs(a-b) for a,b in zip(scores, legacy))
        assert error < 1e-6, (task, scores, legacy)
        report['tasks'][task] = {'checkpoint': settings['checkpoint'], 'scores': scores,
                                  'legacy_max_abs_error': error,
                                  'elapsed_s': time.perf_counter()-t0}
        print(task, report['tasks'][task], flush=True)
        del evaluator
        last = settings
    encoder = LocalSentenceEncoder(last['encoder'], 'cpu')
    units = [Experience('memory', 'wmt19_en_zh', 'test', 'The meeting begins at nine.',
                        '会议九点开始。', '会议九点开始。', 'Preserve the start time.', '',
                        'better', '', '', True, 1., 'test', 'helped')]
    retriever = HybridExperienceRetriever(units, encoder)
    result = retriever.retrieve('The meeting starts at nine.', '会议九点开始。', exclude_source='The meeting starts at nine.')
    assert result.exp_ids == ['memory'], result
    report['hybrid'] = {'retrieved_ids': result.exp_ids, 'input_cosine': result.sim_input,
                         'state_cosine': result.sim_state}
    print('hybrid', report['hybrid'], flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
