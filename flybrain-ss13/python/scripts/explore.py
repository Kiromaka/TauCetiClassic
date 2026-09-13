#!/usr/bin/env python3
"""Разбор коннектома без запуска симуляции.

Когда calibrate.py показывает пустую таблицу, вопрос не «доходит ли сигнал»,
а «почему не доходит». Здесь ответы:

    python3 scripts/explore.py --vocab
        что вообще написано в аннотациях этого снапшота: какие бывают
        super_class, class, типы клеток внутри нисходящих и сенсорных.
        С этого стоит начинать, если популяции разрешились в «НЕ НАЙДЕНО»
        или в «откат на весь класс».

    python3 scripts/explore.py --grep "DN[ap]"
        найти типы клеток по шаблону — проверить гипотезу об именовании.

    python3 scripts/explore.py --stats
        связность каждой популяции: исходящая степень, суммарный вес.
        Популяция без исходящих связей не может ничего возбудить,
        сколько её ни накачивай.

    python3 scripts/explore.py --paths
        весовая достижимость сенсорика -> нисходящие за N шагов.
        Пустая строка означает, что пути нет вовсе, и крутить усиление
        бесполезно: надо менять точку входа.

    python3 scripts/explore.py --inputs forward
        кто на самом деле кормит эту команду. Прямой ответ на вопрос
        «что дёргать, чтобы она заработала».
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flybrain import analysis as A                     # noqa: E402
from flybrain.config import Config                     # noqa: E402
from flybrain.connectome import Connectome, make_synthetic  # noqa: E402
from flybrain.populations import resolve               # noqa: E402


# Ключевые слова, по которым ищем кандидатов в словаре снапшота.
HINTS = {
    "gust_sugar":     ["gr5a", "gr64", "sugar", "sweet", "sgrn"],
    "gust_bitter":    ["gr66", "gr33", "bitter", "bgrn"],
    "gust_water":     ["ppk28", "water", "wgrn"],
    "olf_attractive": ["orn_dm", "orn_va", "orn_dc", "or42", "or92"],
    "olf_aversive":   ["orn_da2", "or56", "geosmin", "orn_dl"],
    "olf_co2":        ["gr21", "gr63", "orn_v", "co2"],
    "thermo_hot":     ["hc", "hot", "warm", "trpa", "ac neuron"],
    "thermo_cold":    ["cc", "cold", "cool"],
    "wind_jo":        ["jo-", "johnston", "jons"],
    "bristle":        ["bristle", "bm_", "chordotonal", "campaniform"],
    "feed":           ["mn9", "mn11", "mn12", "proboscis", "pharyn", "rostrum",
                       "haustell", "fdg", "e49", "motor"],
    "escape":         ["dnp01", "giant fiber", "gf"],
    "forward":        ["dnp09", "dnb0", "dng1"],
    "backward":       ["mdn", "moonwalker"],
    "turn_left":      ["dna0", "dna1", "dnae"],
    "turn_right":     ["dna0", "dna1", "dnae"],
    "groom":          ["dng11", "dng12", "groom"],
    "threat":         ["dnp02", "dnp04", "dnp11", "dnp10"],
    "speed":          ["dna08", "dng13", "dnb"],
}


def suggest(cx, pops, cfg) -> None:
    """Предложить шаблоны для каналов, которые не нашлись или схлопнулись."""
    import collections
    import json
    import re as _re

    vocab = collections.Counter()
    for t, h in zip(cx.cell_type, cx.hb_type):
        if t:
            vocab[t] += 1
        elif h:
            vocab[h] += 1

    broken = []
    seen = collections.defaultdict(list)
    for store, kind in ((pops.sensory, "sensory"), (pops.motor, "motor")):
        for name, pop in store.items():
            if name in ("dn_all", "dn_left", "dn_right"):
                continue
            if len(pop) == 0:
                broken.append((kind, name, "не найдено"))
            elif "откат" in (pop.note or ""):
                broken.append((kind, name, "откат на весь класс"))
            if len(pop):
                seen[pop.idx.tobytes()].append(name)
    for names in seen.values():
        if len(names) > 1:
            for n in names:
                if not any(n == b[1] for b in broken):
                    kind = "sensory" if n in pops.sensory else "motor"
                    broken.append((kind, n, "схлопнулось с " +
                                   ", ".join(x for x in names if x != n)))

    print("\n" + "=" * 70)
    print("ПРЕДЛОЖЕНИЯ ПО ШАБЛОНАМ")
    print("=" * 70)
    if not broken:
        print("Все каналы разрешились нормально, править нечего.")
        return

    fragment = {"sensory": {}, "motor": {}}
    for kind, name, why in broken:
        print(f"\n{name}  ({why})")
        hints = HINTS.get(name, [])
        found = []
        for h in hints:
            rx = _re.compile(_re.escape(h), _re.IGNORECASE)
            for t, c in vocab.items():
                if rx.search(t):
                    found.append((t, c))
        found = sorted(set(found), key=lambda p: -p[1])[:12]
        if found:
            print("    подходящие типы в этом снапшоте: "
                  + ", ".join(f"{t}({c})" for t, c in found))
            pattern = "^(" + "|".join(_re.escape(t) for t, _ in found) + ")$"
            spec = dict(cfg.sensory[name].__dict__ if kind == "sensory"
                        else cfg.motor[name].__dict__)
            spec["match_type"] = pattern
            spec = {k: v for k, v in spec.items()
                    if v not in (None, "", [], 0, False)}
            fragment[kind][name] = spec
        else:
            print(f"    ничего похожего на {hints} в словаре нет.")
            print("    Посмотрите глазами: --vocab, затем --grep '<своё>'")

    if fragment["sensory"] or fragment["motor"]:
        print("\nГотовый кусок для flybrain.json:")
        print(json.dumps({k: v for k, v in fragment.items() if v},
                         ensure_ascii=False, indent=2))
        print("\nПосле правки: python3 scripts/calibrate.py --sweep")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connectome")
    ap.add_argument("--config")
    ap.add_argument("--synthetic", type=int, nargs="?", const=30000)
    ap.add_argument("--vocab", action="store_true")
    ap.add_argument("--grep", metavar="RE")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--paths", action="store_true")
    ap.add_argument("--suggest", action="store_true",
                    help="предложить шаблоны для ненайденных и схлопнувшихся каналов")
    ap.add_argument("--inputs", metavar="ПОПУЛЯЦИЯ")
    ap.add_argument("--outputs", metavar="ПОПУЛЯЦИЯ")
    ap.add_argument("--hops", type=int, default=6)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    cfg = Config.load(a.config)
    print(f"конфиг: {cfg.loaded_from or 'не найден, всё по умолчанию'}")
    if a.synthetic:
        cx = make_synthetic(a.synthetic, 108, params=cfg.model)
    else:
        path = a.connectome or cfg.connectome_file
        if not os.path.exists(path):
            print(f"нет файла {path}; сначала scripts/build_connectome.py",
                  file=sys.stderr)
            return 1
        cx = Connectome.load(path)
    print(cx.describe())

    if not any([a.vocab, a.grep, a.stats, a.paths, a.inputs, a.outputs, a.suggest]):
        a.vocab = a.stats = a.paths = True     # по умолчанию — полный разбор

    pops = resolve(cx, cfg.sensory, cfg.motor)
    sens = {k: v.idx for k, v in pops.sensory.items() if len(v)}
    motor = {k: v.idx for k, v in pops.motor.items() if len(v) and k != "dn_all"}

    if a.vocab:
        print(A.vocabulary(cx, top=a.top))

    if a.suggest:
        suggest(cx, pops, cfg)

    if a.grep:
        rows = A.grep_types(cx, a.grep)
        print(f"\nтипы по шаблону {a.grep!r}: {len(rows)} совпадений")
        for t, c in rows:
            print(f"    {c:>6}  {t}")
        if not rows:
            print("    ничего. Попробуйте --vocab, чтобы увидеть словарь целиком.")

    if a.stats or a.paths or a.inputs or a.outputs:
        print("\nстрою транспонированный вид коннектома…", flush=True)
        cscv = A.csc(cx)

    if a.stats:
        print("\n" + A.STATS_HEADER)
        print("-" * len(A.STATS_HEADER))
        cache = {}
        for group in (sens, motor):
            for name, idx in group.items():
                print(A.pop_stats(cx, name, idx, cscv, cache).line())
            print()
        print("исх.вес — суммарный |вес| исходящих связей на нейрон, в мВ.")
        print("Если он около нуля, популяция физически не может никого возбудить:")
        print("накачивать её бесполезно, нужна другая точка входа.")

    if a.paths:
        print(f"\nвесовая достижимость сенсорика -> нисходящие, до {a.hops} шагов")
        print("в клетке: доля влияния и номер шага, на котором она максимальна\n")
        table = A.reach_table(cx, sens, motor, hops=a.hops)
        names = list(motor)
        print(f"{'':<18}" + "".join(f"{n[:9]:>12}" for n in names))
        print("-" * (18 + 12 * len(names)))
        for sname, row in table.items():
            cells = []
            for n in names:
                v, h = row[n]
                cells.append(f"{'.':>12}" if v <= 0 else f"{v:>8.1e}@{h}")
            print(f"{sname:<18}" + "".join(cells))
        print("\nТочка — пути нет вообще: усиление тут не поможет, меняйте")
        print("точку входа (см. --inputs <команда>).")

    for flag, fn, title in ((a.inputs, A.top_inputs, "входы"),
                            (a.outputs, A.top_outputs, "выходы")):
        if not flag:
            continue
        pool = {**sens, **motor}
        if flag not in pool:
            print(f"\nнет популяции {flag!r}. Есть: {', '.join(sorted(pool))}",
                  file=sys.stderr)
            continue
        rows = (fn(cx, pool[flag], k=a.top, cscv=cscv)
                if fn is A.top_inputs else fn(cx, pool[flag], k=a.top))
        print(f"\n{title} популяции {flag} ({len(pool[flag])} нейронов), "
              f"топ {a.top} по суммарному весу:")
        print(f"    {'тип клетки':<28}{'вес, мВ':>10}{'нейронов':>10}")
        for t, w, c in rows:
            print(f"    {t[:28]:<28}{w:>10.1f}{c:>10}")
        if not rows:
            print("    пусто — у популяции нет связей в эту сторону")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
