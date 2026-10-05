"""
Check that every third-party module app.py imports is listed in requirements.txt.

  python check_requirements.py      # exit 1 if anything is missing

Follows imports into local modules (e.g. collect.py), including imports made
inside functions, so lazily imported packages like lightgbm are covered too.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY = "app.py"
DIST = {}                    # import name -> pip name, where they differ
IMPLICIT = {"pyarrow"}       # not imported by name: pandas.read_parquet engine


def imports(path, seen):
    if path in seen:
        return set()
    seen.add(path)
    out = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            out.add(n.module.split(".")[0])
    for m in list(out):
        local = ROOT / f"{m}.py"
        if local.exists():
            out.discard(m)
            out |= imports(local, seen)
    return out


def main():
    reqs = {re.split(r"[<>=!~;\[ ]", l.strip())[0].lower().replace("_", "-")
            for l in (ROOT / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.lstrip().startswith("#")}
    third = sorted(imports(ROOT / ENTRY, set()) - set(sys.stdlib_module_names))
    need = {DIST.get(m, m).lower().replace("_", "-") for m in third} | IMPLICIT
    missing = sorted(need - reqs)
    for pkg in sorted(need):
        print(f"  {pkg:12s} {'ok' if pkg in reqs else 'MISSING'}")
    if missing:
        print(f"missing from requirements.txt: {', '.join(missing)}")
        sys.exit(1)
    print("requirements.txt covers every import in", ENTRY)


if __name__ == "__main__":
    main()
