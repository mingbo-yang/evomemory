from pathlib import Path
import os,signal,subprocess,sys
from laya_relaxed_flow import OUT,ROOT
from semantic_label_common import dump


def main():
    for stage,script in [('bleu_control','laya_relaxed_bleu_control.py'),('combined_report','report_laya_relaxed_flow.py')]:
        with (OUT/f'{stage}.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            dump(OUT/'bleu_control_job_status.json',{'stage':stage,'pid':p.pid,'process_group':p.pid})
            try:code=p.wait()
            finally:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if code:raise RuntimeError(f'{stage} failed with code {code}')
    dump(OUT/'bleu_control_job_status.json',{'stage':'complete'})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(OUT/'bleu_control_job_status.json',{'stage':'failed','error':repr(exc)});raise
