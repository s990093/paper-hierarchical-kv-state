"""Build every stage-2 report figure from results/ (CPU only).

Usage (repo root):
    nice -n 19 /mlsteam/workspace/venv/tiara/bin/python code/report_stage2/make_figs.py [--only F03 F05 ...]
Outputs: docs/precision_20261013/report_stage2/figs/<id>.{pdf,png} and figs/SOURCES.md
"""
import argparse
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (blocks torch, sets paths)

MODULES = ["figs_main", "figs_explore_a", "figs_explore_b", "figs_report_only"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="figure id prefixes, e.g. F03 F12")
    a = ap.parse_args()
    common.setup()
    failed = []
    for mod in MODULES:
        try:
            M = importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name == mod:
                print(f"[skip] module {mod} not present")
                continue
            raise
        for fn in M.ALL:
            name = fn.__name__  # convention: fNN_<slug>, saved as FNN_<slug>.{pdf,png}
            if a.only and not any(name.lower().startswith(o.lower()) for o in a.only):
                continue
            try:
                out = fn()
                for p, sz in out or []:
                    print(f"[ok] {os.path.relpath(p, common.REPO)}  {sz/1024:.0f} KB")
            except Exception:
                failed.append(name)
                traceback.print_exc()
    common.write_sources_merge()
    if failed:
        print("FAILED:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
