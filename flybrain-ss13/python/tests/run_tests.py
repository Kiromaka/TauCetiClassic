#!/usr/bin/env python3
"""Крошечный раннер, чтобы тесты гонялись без pytest.

    python3 tests/run_tests.py            # всё
    python3 tests/run_tests.py lif        # только test_lif.py

Если pytest установлен, файлы тестов работают и под ним — они обычные
функции test_*.
"""
import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)


def main(argv) -> int:
    want = argv[1:] or None
    files = sorted(f[:-3] for f in os.listdir(HERE)
                   if f.startswith("test_") and f.endswith(".py"))
    if want:
        files = [f for f in files if any(w in f for w in want)]
    ok = fail = 0
    failures = []
    for modname in files:
        mod = importlib.import_module(modname)
        print(f"\n\033[1m{modname}\033[0m")
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, name)
            t0 = time.perf_counter()
            try:
                fn()
                dt = (time.perf_counter() - t0) * 1000
                print(f"  \033[32m✓\033[0m {name}  ({dt:.0f} мс)")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  \033[31m✗\033[0m {name}: {exc}")
                failures.append((modname, name, traceback.format_exc()))
                fail += 1
    print(f"\n{'='*54}\n{ok} прошло, {fail} упало")
    for m, n, tb in failures:
        print(f"\n--- {m}.{n} ---\n{tb}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
