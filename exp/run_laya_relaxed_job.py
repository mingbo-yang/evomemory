"""Run the current binary rollout in its own process group; no threshold/report-v1 fallback."""
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('laya_relaxed_flow.py')), *sys.argv[1:]],
                               start_new_session=True)
    try:
        code = process.wait()
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(code)


if __name__ == '__main__':
    main()
