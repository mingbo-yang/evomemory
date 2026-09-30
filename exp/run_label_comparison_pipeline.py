"""Durable sequential training/evaluation after annotation releases the GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from semantic_label_common import OUT,ROOT,dump


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'


def run():
    job=json.loads((OUT/'relabel_job.json').read_text())
    while alive(job['pid']):time.sleep(10)
    state=json.loads((OUT/'train_label_status.json').read_text())
    assert state.get('state')=='complete','Training annotation failed: '+str(job)
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free>=14000:break
        time.sleep(10)
    for stage,script in [('training','train_label_comparison.py'),('evaluation','evaluate_label_comparison.py')]:
        dump(OUT/'pipeline_status.json',{'stage':stage,'pid':os.getpid()})
        with (OUT/f'{stage}.log').open('a') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    dump(OUT/'pipeline_status.json',{'stage':'complete','report':str(OUT/'COMPARISON_ZH.md')})


if __name__=='__main__':
    try:run()
    except Exception as exc:
        dump(OUT/'pipeline_status.json',{'stage':'failed','error':repr(exc)})
        raise
