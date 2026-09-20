#!/usr/bin/env python3
"""Run the suite without pytest.

`pytest` is the normal entry point. This exists so the tests are runnable in
a bare environment, and because the suite was written with plain functions
rather than fixtures precisely so that both routes work.

    python tests/run.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODULES = [
    "tests.test_pincode",
    "tests.test_demo_profiles",
    "tests.test_model_behaviour",
    "tests.test_policy_and_edges",
    "tests.test_schemas",
]


def main() -> int:
    passed, failed, skipped = 0, [], 0

    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            print(f"SKIP  {module_name} ({exc})")
            skipped += 1
            continue

        print(f"\n{module_name}")
        for name in sorted(vars(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception:
                failed.append((module_name, name, traceback.format_exc()))
                print(f"  FAIL  {name}")
            else:
                passed += 1
                print(f"  ok    {name}")

    print("\n" + "=" * 68)
    for module_name, name, tb in failed:
        print(f"\nFAILED {module_name}::{name}\n{tb}")
    print(f"{passed} passed, {len(failed)} failed, {skipped} module(s) skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
