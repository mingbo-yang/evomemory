"""Durable single-job launcher for the approved acceptance collection."""
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs/acceptance_data_v2'

def status(value):
    value['updated_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    temp = OUT/'job_status.json.tmp'
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(OUT/'job_status.json')

if __name__ == '__main__':
    lock = (OUT/'job.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [sys.executable, '-u', str(ROOT/'acceptance_data.py'), '--folds',
               'train', 'development', 'temperature_calibration', 'threshold_calibration', 'test']
    metadata = {'state': 'starting', 'pid': os.getpid(), 'command': command,
                'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'code_sha256': hashlib.sha256((ROOT/'acceptance_data.py').read_bytes()).hexdigest(),
                'versions': {p: version(p) for p in ['torch','transformers','vllm','sentence-transformers']}}
    (OUT/'collector_at_full_launch.py.txt').write_bytes((ROOT/'acceptance_data.py').read_bytes())
    status(metadata)
    with (OUT/'full_collection.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status({**metadata, 'state':'running', 'collector_pid':child.pid})
        code = child.wait()
    if code:
        status({**metadata, 'state':'failed', 'exit_code':code})
        sys.exit(code)
    report = json.loads((OUT/'audit_summary.json').read_text())
    counts = report['folds']
    complete = counts['train']['unique_valid_pairs'] >= 20000
    complete &= all(counts[f]['sources_completed'] == n for f,n in
                    [('development',400),('temperature_calibration',256),('threshold_calibration',256),('test',400)])
    status({**metadata, 'state':'collected_pending_label_margin' if complete else 'insufficient_data',
            'exit_code':code, 'audit': report})
    sys.exit(0 if complete else 2)
