"""Run the authorized rollout and evaluation; clean up only its own GPU workers."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import signal
import subprocess
import sys
from laya_relaxed_flow import OUT,ROOT
from semantic_label_common import dump


def main():
    for stage,script in [('rollout','laya_relaxed_flow.py'),('report','report_laya_relaxed_flow.py')]:
        with (OUT/f'{stage}.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            dump(OUT/'job_status.json',{'stage':stage,'pid':process.pid,'process_group':process.pid,
                 'updated_at_utc':datetime.now(timezone.utc).isoformat()})
            try:code=process.wait()
            finally:
                # Each stage has a private session; unrelated jobs cannot be in it.
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if code!=0:raise RuntimeError(f'{stage} exited {code}; see {OUT/stage}.log')
    dump(OUT/'job_status.json',{'stage':'complete','report':str(OUT/'RESULTS_ZH.md')})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(OUT/'job_status.json',{'stage':'failed','error':repr(exc)});raise
