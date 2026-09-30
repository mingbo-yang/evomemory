"""Build a source-only ZIP using an allowlist; never read local credentials."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import zipfile


def package_source(root: Path, output: Path) -> dict:
    root = root.resolve()
    package = root / "evoscope"
    candidates = [root / "README.md"]
    for pattern in ("*.py", "*.md", "requirements*.txt"):
        candidates.extend(package.glob(pattern))
    candidates.extend((package / "docs").rglob("*.md"))
    candidates.extend((package / "tests").rglob("*.py"))
    candidates.extend((package / "scripts").glob("*.sh"))
    candidates.extend(package / "examples" / name for name in
                      ("initial_policies.json", "manifest.template.json"))
    candidates.append(package / ".gitignore")
    entries = {}
    for path in sorted(set(candidates)):
        relative = path.relative_to(root)
        if any(part in {".local", ".git", "__pycache__", "results"} or part.startswith(".env")
               for part in relative.parts):
            continue
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents) if p != root):
            continue
        if not path.resolve().is_relative_to(root):
            continue
        content = path.read_bytes()
        if re.search(rb"sk-[A-Za-z0-9_-]{16,}", content):
            raise ValueError(f"possible API credential in allowed source file: {relative}")
        entries[str(relative)] = content
    manifest = {"format": "source-only", "excluded": [".local/", ".env*", "results/", "archives", "virtualenvs"],
                "files": [{"path": name, "sha256": sha256(content).hexdigest()}
                          for name, content in entries.items()]}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never silently replace an earlier release.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr("evomemory/" + name, content)
        archive.writestr("evomemory/PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
    return {"output": str(output.resolve()), "source_files": len(entries),
            "sha256": sha256(output.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(package_source(Path(__file__).resolve().parent.parent, Path(args.output)), indent=2))


if __name__ == "__main__":
    main()
