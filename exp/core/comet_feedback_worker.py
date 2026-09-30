"""Local COMET worker. Invoked only by the offline/delayed feedback adapter."""
import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    if not args.checkpoint.is_file() or args.output.exists():
        raise ValueError('Checkpoint must exist and output must be new')
    from comet import load_from_checkpoint
    model = load_from_checkpoint(str(args.checkpoint))
    examples = json.loads(args.input.read_text())
    result = model.predict(examples, batch_size=args.batch_size, gpus=int(args.device == 'cuda'), num_workers=0)
    scores = [float(v) for v in (result.scores if hasattr(result, 'scores') else result['scores'])]
    if len(scores) != len(examples) or any(not math.isfinite(v) for v in scores):
        raise ValueError('COMET returned invalid scores')
    args.output.write_text(json.dumps(scores, allow_nan=False))


if __name__ == '__main__':
    main()
