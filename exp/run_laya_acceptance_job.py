"""Run training then frozen calibration/test, preserving failure states and logs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OUT=Path('/mnt/huawei/ymb/model/laya-multilingual-acceptance-enzh-v1')

def write(value):
    value['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    p=OUT/'job_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

if __name__=='__main__':
    lock=(OUT/'job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((OUT/'smoke_result.json').read_text())['weights_updated']
    assert json.loads((OUT/'adapter_preflight.json').read_text())['pass']
    for name in ['laya_acceptance_common.py','train_laya_acceptance.py','evaluate_laya_acceptance.py']:
        (OUT/(name+'.snapshot')).write_bytes((ROOT/name).read_bytes())
    for stage,script in [('training','train_laya_acceptance.py'),('validation','evaluate_laya_acceptance.py')]:
        with (OUT/f'{stage}.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write({'state':'running','stage':stage,'supervisor_pid':os.getpid(),'child_pid':proc.pid})
            code=proc.wait()
        if code:
            write({'state':'failed','stage':stage,'exit_code':code})
            sys.exit(code)
    report=json.loads((OUT/'evaluation_report.json').read_text())
    write({'state':'training_and_verifier_validation_complete','verifier_gate_passed':report['verifier_gate_passed'],
           'next_stage':report['next_stage'],'checkpoint':str(OUT/'best'),'exit_code':0})
