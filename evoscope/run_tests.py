"""Run with an isolated temporary root (works on shared servers)."""
from pathlib import Path
import tempfile

import pytest


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="evoscope-tests-") as root:
        return pytest.main([str(Path(__file__).parent / "tests"), "-q", "--tb=short",
                            "--basetemp", str(Path(root) / "pytest")])


if __name__ == "__main__":
    raise SystemExit(main())
