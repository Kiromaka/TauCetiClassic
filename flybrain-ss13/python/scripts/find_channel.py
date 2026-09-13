#!/usr/bin/env python3
"""Поиск нейронов, специфично управляемых одним сенсорным каналом.

Зачем это нужно. На настоящей выгрузке FlyWire пищевой канал нашёлся из
ТРЁХ нейронов, и то generic-расширением шаблона: настоящих MN9 в снапшоте
нет вовсе, класса motor в super_class тоже нет. Три нейрона насыщаются от
любого входа, поэтому в калибровке feed отвечал +250 Гц и на сахар, и на
воду, и на стену. Это выглядит как работающий мозг, но мозгом не является.

Скрипт отвечает на вопрос "а есть ли в этих данных нейроны, которые тянет
ИМЕННО сахар и не тянет всё остальное". Считается по графу, симуляция не
запускается:

    для каждого нейрона: влияние выбранного канала минус максимум влияния
    всех прочих сенсорных каналов

Высокий счёт значит "этот нейрон слушает почти только сахар". Дальше можно
взять верхушку списка как пищевую популяцию — скрипт печатает готовую
строку для flybrain.json.

    python3 scripts/find_channel.py --for feed --from gust_sugar
    python3 scripts/find_channel.py --for feed --from gust_sugar --apply

Ярлык при этом остаётся гипотезой: данные говорят "эти нейроны слушают
сахар", а не "эти нейроны командуют хоботком".
"""
import argparse
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flybrain.analysis import influence                    # noqa: E402
from flybrain.config import Config                         # noqa: E402
from flybrain.connectome import Connectome, make_synthetic  # noqa: E402
from flybrain.populations import resolve                    # noqa: E402


def specificity(cx, pops, target_channel: str, hops: int = 3) -> Dict[str, np.ndarray]:
    """Влияние целевого канала и максимум влияния всех остальных."""
    inf: Dict[str, np.ndarray] = {}
    for name, pop in pops.sensory.items():
        if len(pop) == 0:
            continue
        steps = influence(cx, pop.idx, hops=hops)
        # суммируем шаги 1..hops: нас интересует "доходит ли вообще",
        # а не на каком именно шаге
        inf[name] = np.sum(steps[1:], axis=0)
    return inf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connectome")
    ap.add_argument("--config")
    ap.add_argument("--synthetic", type=int, nargs="?", const=20000)
    ap.add_argument("--for", dest="channel", default="feed",
                    help="какой моторный канал ищем")
    ap.add_argument("--from", dest="source", default="gust_sugar",
                    help="какой сенсорный канал должен его тянуть")
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--among", default="descending|motor",
                    help="regex по super_class: где искать кандидатов")
    ap.add_argument("--margin", type=float, default=1.4,
                    help="во сколько раз свой канал должен быть сильнее")
    ap.add_argument("--apply", action="store_true",
                    help="записать найденные типы в flybrain.json")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    print(f"конфиг: {cfg.loaded_from or 'не найден, всё по умолчанию'}")
    if a.synthetic:
        cx = make_synthetic(a.synthetic, 108, params=cfg.model)
        print(f"синтетический коннектом на {a.synthetic} нейронов")
    else:
        path = a.connectome or cfg.connectome_file
        cx = Connectome.load(path)
        print(f"коннектом: {path}")
    pops = resolve(cx, cfg.sensory, cfg.motor)

    if a.source not in pops.sensory or len(pops.sensory[a.source]) == 0:
        print(f"сенсорный канал {a.source} пуст — искать нечем")
        return 1

    print(f"\nсчитаю влияние {len(pops.sensory)} сенсорных каналов "
          f"на {a.hops} шага…", flush=True)
    inf = specificity(cx, pops, a.channel, hops=a.hops)
    mine = inf[a.source]
    others = np.zeros_like(mine)
    for name, v in inf.items():
        if name == a.source:
            continue
        np.maximum(others, v, out=others)

    import re
    cand = np.flatnonzero(
        np.array([bool(re.search(a.among, s or "", re.I)) for s in cx.super_class]))
    if cand.size == 0:
        print(f"по super_class ~ /{a.among}/ нет ни одного нейрона; "
              f"есть: {sorted(set(cx.super_class))[:10]}")
        return 1

    # Группируем по ТИПУ: пользователю нужен шаблон для конфига, а не
    # двадцать пять root_id. Плюс тип из одного специфичного нейрона и
    # тип, специфичный целиком, — это очень разные находки.
    import collections
    import re
    by_type: Dict[str, list] = collections.defaultdict(list)
    for i in cand:
        t = str(cx.cell_type[i] or cx.hb_type[i] or "")
        by_type[t or "(без типа)"].append(i)

    rows = []
    for t, ids in by_type.items():
        ids_a = np.array(ids)
        m = float(mine[ids_a].mean())
        o = float(others[ids_a].mean())
        # Специфичным считаем нейрон, у которого свой канал заметно сильнее
        # всех прочих, а не просто на волосок: иначе в список попадает
        # каждый второй нейрон мозга.
        spec = int(np.count_nonzero(mine[ids_a] > others[ids_a] * a.margin))
        rows.append((t, len(ids), m, o, m / max(o, 1e-12), spec))
    rows.sort(key=lambda r: -r[4])

    print(f"кандидатов в /{a.among}/: {cand.size}, типов: {len(rows)}\n")
    print(f"{'тип':<24}{'нейронов':>9}{'свой':>11}{'чужой':>11}"
          f"{'во сколько':>12}{'специфичных':>13}")
    print("-" * 82)
    for t, cnt, m, o, ratio, spec in rows[:a.top]:
        print(f"{t[:24]:<24}{cnt:>9}{m:>11.3g}{o:>11.3g}"
              f"{ratio:>12.2f}{spec:>8}/{cnt:<4}")

    # Берём типы, у которых специфично БОЛЬШИНСТВО нейронов и отношение
    # выше порога: один случайно попавший нейрон канала не делает.
    good = [(t, cnt, ratio) for t, cnt, m, o, ratio, spec in rows
            if ratio >= a.margin and spec * 2 >= cnt and t != "(без типа)"]
    print(f"\nтипов, специфичных большинством нейронов: {len(good)}")
    if not good:
        print(f"\nВ этих данных нет типа, который слушает {a.source} хотя бы в "
              f"{a.margin:g} раз сильнее всего остального.")
        print(f"Значит команды «{a.channel}» в выгрузке нет, и честнее "
              "оставить канал ненайденным:")
        print('  рефлекторная дуга отработает поведение, а в dbg["source"] '
              'будет стоять')
        print("  её имя, а не «мозг» — то есть на дашборде сразу видно, "
              "что решает не коннектом.")
        return 0

    types = [t for t, _, _ in good]
    for t, cnt, ratio in good:
        print(f"  {t:<22} {cnt:>4} нейронов, свой канал сильнее в "
              f"{ratio:.1f} раз")
    pattern = "^(" + "|".join(re.escape(t) for t in types) + ")$"
    print("\nПрописать в flybrain.json:")
    print(f'    "motor": {{ "{a.channel}": {{ "match_type": "{pattern}" }} }}')
    if a.apply:
        where = cfg.patch_file({"motor": {a.channel: {"match_type": pattern}}})
        print(f"Записал в {where}. Перезапустите сервер.")
    else:
        print("Флаг --apply запишет это сам.")
    print("\nЯрлык остаётся гипотезой: данные говорят «эти нейроны слушают "
          f"{a.source}»,")
    print(f"а не «эти нейроны командуют {a.channel}». Проверять — "
          "калибровкой и поведенческой пробой.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
