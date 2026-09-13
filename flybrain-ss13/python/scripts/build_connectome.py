#!/usr/bin/env python3
"""Собрать data/connectome.npz из скачанных CSV.

Делается один раз: разбор ~15 миллионов строк занимает несколько минут,
дальше демон поднимается за секунды.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flybrain.config import Config                                   # noqa: E402
from flybrain.connectome import (build, columns_of,                  # noqa: E402
                                 find_type_sources)


def inspect(data_dir: str) -> int:
    """Показать, что лежит в каталоге данных и какие там колонки."""
    if not os.path.isdir(data_dir):
        print(f"нет каталога {data_dir}", file=sys.stderr)
        return 1
    files = sorted(f for f in os.listdir(data_dir)
                   if f.endswith(".csv") or f.endswith(".csv.gz"))
    if not files:
        print(f"в {data_dir} нет ни одного csv")
        return 1
    for name in files:
        path = os.path.join(data_dir, name)
        size = os.path.getsize(path) / 1e6
        try:
            cols = columns_of(path)
        except Exception as exc:
            print(f"{name}  ({size:.0f} МБ)  — не прочитался: {exc}")
            continue
        print(f"{name}  ({size:.0f} МБ)")
        print(f"    {cols}")
    print("\nисточники типов клеток: "
          + (", ".join(os.path.basename(p) for p in find_type_sources(data_dir))
             or "НЕ НАЙДЕНЫ"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/connectome.npz")
    ap.add_argument("--gain", type=float, default=None,
                    help="глобальный множитель веса синапса (по умолчанию 0.65)")
    ap.add_argument("--min-syn", type=int, default=None)
    ap.add_argument("--config")
    ap.add_argument("--inspect", action="store_true",
                    help="только показать, какие файлы и колонки есть в каталоге")
    a = ap.parse_args()

    if a.inspect:
        return inspect(a.data)

    cfg = Config.load(a.config)
    if a.gain is not None:
        cfg.model.gain = a.gain
    if a.min_syn is not None:
        cfg.model.min_syn_count = a.min_syn

    def path(name):
        for ext in (".gz", ""):
            p = os.path.join(a.data, name + ext)
            if os.path.exists(p):
                return p
        return None

    conn = path("connections.csv")
    clas = path("classification.csv")
    if not conn or not clas:
        print("не нашёл connections.csv(.gz) и/или classification.csv(.gz) в "
              f"{a.data}. Сначала: python3 scripts/fetch_connectome.py",
              file=sys.stderr)
        return 1

    # Типы клеток от снапшота к снапшоту переезжают между файлами, поэтому
    # берём в качестве источников все CSV каталога, где есть root_id и
    # колонка типа. Порядок — от самых подходящих по имени.
    type_csvs = find_type_sources(a.data)
    if type_csvs:
        print("источники типов клеток: "
              + ", ".join(os.path.basename(p) for p in type_csvs))
    else:
        print("ВНИМАНИЕ: файлов с типами клеток в каталоге не видно.\n"
              "          Скачайте их: python3 scripts/fetch_connectome.py --all",
              file=sys.stderr)

    t0 = time.time()
    try:
        cx = build(conn, clas,
                   coordinates_csv=path("coordinates.csv"),
                   consolidated_csv=path("consolidated_cell_types.csv"),
                   type_csvs=type_csvs,
                   params=cfg.model)
    except (RuntimeError, KeyError) as exc:
        print("\n" + str(exc).strip('"'), file=sys.stderr)
        print("\nЧто лежит в каталоге, покажет:\n"
              f"    python3 scripts/build_connectome.py --data {a.data} --inspect",
              file=sys.stderr)
        return 1
    cx.save(a.out)
    size = os.path.getsize(a.out) / 1e6
    print(f"\nсохранено: {a.out}  ({size:.0f} МБ, {time.time()-t0:.0f} с)")

    # сразу показываем, какие популяции нашлись — если что-то пусто,
    # лучше узнать об этом здесь, а не в игре
    from flybrain.populations import resolve
    print()
    print(resolve(cx, cfg.sensory, cfg.motor).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
