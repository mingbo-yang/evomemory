"""Versioned, local-only configuration for optimized experiments."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from functools import lru_cache
from .quality_feedback import QualityPolicy


@lru_cache(maxsize=64)
def _file_digest(path, size, mtime_ns):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(path):
    p = Path(path).resolve(strict=True)
    st = p.stat()
    return _file_digest(str(p), st.st_size, st.st_mtime_ns)


def resolve_optimization(path, task):
    profile = json.loads(Path(path).read_text())
    if profile.get('version') != 'bert-hybrid-feedback-v1':
        raise ValueError('unsupported optimization profile version')
    settings = dict(profile['tasks'][task])
    policy = {**profile['policy'], 'stop_threshold': settings['stop_threshold']}
    QualityPolicy(**policy)
    retrieval = dict(profile['retrieval'])
    if retrieval.pop('method', 'hybrid') not in ('hybrid', 'bm25'):
        raise ValueError('retrieval method must be hybrid or bm25')
    method = profile['retrieval'].get('method', 'hybrid')
    if set(retrieval) - {'semantic_weight', 'min_semantic_similarity', 'rrf_k'}:
        raise ValueError('unknown retrieval settings')
    if not (0 <= retrieval.get('semantic_weight', .5) <= 1
            and -1 <= retrieval.get('min_semantic_similarity', .2) <= 1
            and retrieval.get('rrf_k', 60) > 0):
        raise ValueError('invalid hybrid retrieval parameters')
    base = str(Path(profile['bert_base']).resolve(strict=True))
    checkpoint = str(Path(settings['checkpoint']).resolve(strict=True))
    # Include content identities rather than relying on mutable paths.
    assets = {'checkpoint': digest(checkpoint)}
    for name in ('config.json', 'vocab.txt', 'tokenizer_config.json'):
        assets['bert/' + name] = digest(Path(base) / name)
    encoder = str(Path(profile['encoder']).resolve(strict=True))
    if method == 'hybrid':
        for file in sorted(Path(encoder).rglob('*')):
            if file.is_file() and file.suffix in ('.json', '.txt', '.safetensors', '.bin'):
                assets['encoder/' + str(file.relative_to(encoder))] = digest(file)
    # CPU defaults avoid changing vLLM's GPU memory budget.
    return dict(version=profile['version'], bert_base=base, checkpoint=checkpoint,
                encoder=encoder, device=profile.get('device', 'cpu'),
                policy=policy, retrieval=dict(method=method, **retrieval),
                assets=assets, feedback='delayed_reference_metric',
                admission='accepted_and_positive_feedback', transition_replay='accepted_state_v2')


def build_components(settings, library):
    from .quality_feedback import BertQualityEvaluator
    from .hybrid_retrieval import LocalSentenceEncoder, HybridExperienceRetriever
    from .bm25_fields import ExperienceRetriever
    evaluator = BertQualityEvaluator(settings['checkpoint'], settings['bert_base'], settings['device'])
    if not library:
        return evaluator, None
    retrieval = dict(settings['retrieval'])
    method = retrieval.pop('method')
    if method == 'bm25':
        return evaluator, ExperienceRetriever(library)
    encoder = LocalSentenceEncoder(settings['encoder'], settings['device'])
    return evaluator, HybridExperienceRetriever(library, encoder, **retrieval)
