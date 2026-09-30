"""Explicit binary training/evaluation job; importing this module never starts work."""
import argparse
from pathlib import Path
import subprocess
import sys
import laya_acceptance_common as C


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=C.DATA)
    p.add_argument('--output', type=Path, default=C.OUT)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    subprocess.run([sys.executable, str(C.ROOT / 'train_laya_acceptance.py'), '--data', str(args.data),
                    '--output', str(args.output), '--device', args.device], check=True)
    subprocess.run([sys.executable, str(C.ROOT / 'evaluate_laya_acceptance.py'), '--checkpoint', str(args.output / 'best'),
                    '--data', str(args.data), '--output', str(args.output / 'development_evaluation.json'),
                    '--device', args.device], check=True)


if __name__ == '__main__':
    main()
