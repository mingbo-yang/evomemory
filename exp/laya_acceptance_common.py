"""Current Laya Accept/Reject task. Scores and references are never model features."""
from contextlib import nullcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor/laya'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
BASE = Path('/mnt/huawei/ymb/model/laya-multilingual')
OUT = Path('/mnt/huawei/ymb/model/laya-multilingual-accept-reject-v1')
DATA = ROOT / 'runs/acceptance_binary_v1'
COLLECTION = ROOT / 'runs/acceptance_data_v2'
SCHEMA = 'laya-accept-reject-v1'
LABELS = ['Accept', 'Reject']
INPUT_FIELDS = ('source', 'current', 'candidate')
MODEL_POPULATION = 'current != candidate; exact string comparison'
NO_OP_RULE = 'current == candidate -> Reject before Laya; audit only'
CRITERIA = {
    'Accept': 'Replace the current answer: the candidate meaningfully improves translation quality.',
    'Reject': 'Keep the current answer: the candidate is equivalent, worse, or offers no meaningful improvement.',
}
QUESTION = {'t': 'choice', 'ins': 'Should the candidate replace the current English-to-Chinese translation?',
            'crit': CRITERIA}
SEED = 42


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def seed_for(*values):
    return int(hashlib.sha256('|'.join(map(str, values)).encode()).hexdigest()[:8], 16)


def model_input(inputs):
    """Whitelist features; ignore all reference, feedback, retrieval and audit fields."""
    value = {key: inputs[key] for key in INPUT_FIELDS}
    if any(not isinstance(v, str) for v in value.values()):
        raise ValueError('Model input fields must be strings')
    return value


def is_noop(inputs):
    """Exact equality only; do not erase whitespace, punctuation or Unicode edits."""
    return inputs['current'] == inputs['candidate']


def require_changed_pairs(rows):
    for row in rows:
        if is_noop(model_input(row['input'])):
            raise ValueError('No-op pairs belong in audit only, not Laya training/evaluation: ' + str(row.get('id', 'unknown')))


def state_text(inputs, swap=False):
    if swap:
        raise ValueError('Binary Reject cannot be reversed without offline feedback; answer swapping is disabled')
    x = model_input(inputs)
    return (f"Task: English-to-Chinese Translation\n\nSource:\n{x['source']}"
            f"\n\nCurrent Answer:\n{x['current']}\n\nCandidate Answer:\n{x['candidate']}")


def semantic_target(label, swap=False):
    if swap:
        raise ValueError('Do not invert a binary Reject label')
    if label not in LABELS:
        raise ValueError('Expected an Accept/Reject training target')
    return LABELS.index(label)


def validate_checkpoint(path):
    cfg = json.loads((Path(path) / 'rl_agent_config.json').read_text())
    if cfg.get('decision_schema') != SCHEMA or cfg.get('labels') != LABELS:
        raise ValueError('A trained Accept/Reject checkpoint is required; three-class/base checkpoints are not runtime-compatible')
    if cfg.get('input_schema') != list(INPUT_FIELDS):
        raise ValueError('Checkpoint model-input schema mismatch')
    if cfg.get('encoder') != 'jhu-clsp/mmBERT-base':
        raise ValueError('Expected laya-multilingual (mmBERT), not English Laya')
    return cfg


def get_tokenizer(path=BASE):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(Path(path) / 'tokenizer', local_files_only=True)


def load_model(path=OUT / 'best', *, initialization=False):
    from laya.common import build_model
    from safetensors.torch import load_file
    path = Path(path)
    cfg = json.loads((path / 'rl_agent_config.json').read_text()) if initialization else validate_checkpoint(path)
    if cfg.get('encoder') != 'jhu-clsp/mmBERT-base':
        raise ValueError('Must use laya-multilingual')
    if initialization and cfg.get('fine_tuned'):
        raise ValueError('Start the binary task from the base multilingual model, not an old fine-tuned task')
    model = build_model(cfg, encoder_dir=str(path / 'encoder'), pretrained=False)
    model.load_state_dict(load_file(str(path / 'model.safetensors')), strict=True)
    model.encoder.config.reference_compile = False
    return model, cfg


class EncodedPairs:
    def __init__(self, rows, tok, allow_swap=False):
        from laya.common import build_sequence
        if allow_swap:
            raise ValueError('Current/Candidate swapping requires recomputed offline labels and is disabled')
        require_changed_pairs(rows)
        self.rows = rows
        self.tok = tok
        self.prefixes = {}
        for order in itertools.permutations(range(2)):
            ids, markers = build_sequence(tok, '', QUESTION, 1024, 256, option_order=list(order), state_ids=[])
            self.prefixes[order] = (ids[:-1], markers)
        self.states = []
        self.max_len = 0
        for row in rows:
            text = state_text(row['input']).replace(tok.mask_token, ' ')
            ids = tok(text, add_special_tokens=False)['input_ids']
            length = max(len(p[0]) for p in self.prefixes.values()) + len(ids) + 1
            if length > 1024:
                raise ValueError(f"Input would be truncated: {row['id']} ({length} tokens)")
            self.states.append([ids])
            self.max_len = max(self.max_len, length)

    def item(self, i, epoch=None, force_order=None):
        row = self.rows[i]
        order = [0, 1]
        if epoch is not None:
            random.Random(seed_for(SEED, row['id'], epoch)).shuffle(order)
        if force_order is not None:
            order = list(force_order)
        if sorted(order) != [0, 1]:
            raise ValueError('Exactly two options are required')
        prefix, markers = self.prefixes[tuple(order)]
        item = {'ids': prefix + self.states[i][0] + [self.tok.sep_token_id],
                'markers': markers, 'qtype': 0, 'order': order, 'row_index': i, 'swapped': False}
        if 'label' in row:
            semantic = semantic_target(row['label'])
            y = order.index(semantic)
            item.update(label=y, target=[float(j == y) for j in range(2)], semantic_label=semantic)
        return item


def pack(items, tok, device='cpu'):
    from laya.common import collate_items
    batch = collate_items([[i] for i in items], tok.pad_token_id)
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}


def forward(model, batch):
    return model(batch['input_ids'], batch['attention_mask'], batch['marker_pos'], batch['marker_mask'], batch['qtype'])


def probabilities(logits, temperature=1.0):
    """Diagnostic probabilities only; the decision is always made from raw logits."""
    import numpy as np
    z = np.asarray(logits, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2 or not np.isfinite(z).all():
        raise ValueError('Expected finite [N, 2] binary logits')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Diagnostic temperature must be positive and finite')
    if not len(z):
        return z.copy()
    z = z / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def decisions(logits):
    import numpy as np
    z = np.asarray(logits, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2 or not np.isfinite(z).all():
        raise ValueError('Expected finite [N, 2] binary logits')
    # Argmax classification with a fixed Reject tie-break, not a calibrated gate.
    return ['Accept' if a > r else 'Reject' for a, r in z]


def infer(model, encoded, batch_size=32, force_order=None):
    import numpy as np
    import torch
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    device = next(model.parameters()).device
    model.eval()
    out = np.zeros((len(encoded.rows), 2), dtype=np.float32)
    order = sorted(range(len(encoded.rows)), key=lambda i: len(encoded.states[i][0]))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            items = [encoded.item(i, force_order=force_order) for i in indices]
            batch = pack(items, encoded.tok, device)
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
            with amp:
                z, _ = forward(model, batch)
            z = z.float().cpu().numpy()
            if z.shape != (len(items), 2):
                raise ValueError('Laya must return exactly two option scores')
            for j, (idx, item) in enumerate(zip(indices, items)):
                for position, semantic in enumerate(item['order']):
                    out[idx, semantic] = z[j, position]
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite Laya output')
    return out


def allowed(row):
    import unicodedata
    x = model_input(row['input'])
    def length(text):
        return len(''.join(unicodedata.normalize('NFKC', text).casefold().split()))
    return bool(x['candidate'].strip()) and not is_noop(x) and length(x['candidate']) <= 1.5 * length(x['current']) + 8


def accepted_mask(rows, logits):
    import numpy as np
    if len(rows) != len(logits):
        raise ValueError('Row/logit count mismatch')
    return np.array([allowed(r) and d == 'Accept' for r, d in zip(rows, decisions(logits))], dtype=bool)


def metrics(rows, logits, delta, diagnostic_temperature=1.0):
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, log_loss, roc_auc_score
    if not rows or len(rows) != len(delta):
        raise ValueError('Nonempty aligned rows and feedback required')
    require_changed_pairs(rows)
    p = probabilities(logits, diagnostic_temperature)
    y = np.array([semantic_target(r['label']) for r in rows])
    pred = np.array([semantic_target(d) for d in decisions(logits)])
    mask = accepted_mask(rows, logits)
    gain = np.asarray(delta, dtype=float)
    if not np.isfinite(gain).all():
        raise ValueError('Nonfinite offline feedback')
    n = int(mask.sum())
    return {'n': len(rows), 'model_population': MODEL_POPULATION, 'labels': LABELS, 'accuracy': float((pred == y).mean()),
            'macro_f1': float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            'accepted': n, 'coverage': n / len(rows),
            'accept_precision': float((y[mask] == 0).mean()) if n else None,
            'mean_accepted_delta': float(gain[mask].mean()) if n else None,
            'net_delta_per_pair': float(gain[mask].sum() / len(rows)),
            'negative_accepts': int((gain[mask] < 0).sum()),
            'diagnostics': {'temperature': diagnostic_temperature,
                            'nll': float(log_loss(y, p, labels=[0, 1])),
                            'brier': float(((p - np.eye(2)[y]) ** 2).sum(1).mean()),
                            'auc_accept': float(roc_auc_score(y == 0, p[:, 0])) if len(set(y)) == 2 else None},
            'runtime_rule': 'raw two-option argmax; exact tie Reject; no probability thresholds'}


def selection_key(result):
    return result['macro_f1'], result['accuracy']


def save_checkpoint(model, cfg, tok, path, label_feedback):
    import torch
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    # The upstream buffer has one entry per question TYPE, not per class.
    state['temperature'] = torch.ones(3)
    save_file(state, str(path / 'model.safetensors'))
    model.encoder.config.save_pretrained(path / 'encoder')
    tok.save_pretrained(path / 'tokenizer')
    cfg = dict(cfg, max_len=1024, head_max_len=256, temperature=[1., 1., 1.],
               fine_tuned=True, model_name='laya-multilingual-accept-reject-v1',
               decision_schema=SCHEMA, labels=LABELS, input_schema=list(INPUT_FIELDS),
               model_population=MODEL_POPULATION, no_op_rule=NO_OP_RULE,
               decision_rule='argmax; exact tie Reject', label_feedback=label_feedback)
    cfg.pop('temperature_by_options', None)
    dump(path / 'rl_agent_config.json', cfg)


def load_fold(fold, data=DATA):
    data = Path(data)
    meta = json.loads((data / 'dataset.json').read_text())
    if meta.get('decision_schema') != SCHEMA or meta.get('label_order') != LABELS:
        raise ValueError('Expected the COMET-labelled binary dataset, not old three-class data')
    if meta.get('model_population') != MODEL_POPULATION:
        raise ValueError('Re-export changed-only data; no-op pairs must stay in audit sidecars')
    for name in (f'{fold}.jsonl', f'audit/{fold}.jsonl'):
        if digest(data / name) != meta['output_hashes'][name]:
            raise ValueError('Dataset file changed after preparation: ' + name)
    rows = read_jsonl(data / f'{fold}.jsonl')
    if not rows:
        raise ValueError('No changed pairs in split ' + fold)
    audit = {r['id']: r for r in read_jsonl(data / 'audit' / f'{fold}.jsonl')}
    for row in rows:
        if set(row) != {'id', 'input', 'label'} or set(row['input']) != set(INPUT_FIELDS):
            raise ValueError('Unexpected model training fields')
        semantic_target(row['label'])
    require_changed_pairs(rows)
    return rows, [audit[r['id']]['delta'] for r in rows], [audit[r['id']]['sample_id'] for r in rows]
