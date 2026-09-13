#!/usr/bin/env python3
"""Поведенческая проба: работает ли коннектом на самом деле.

Вопрос «муха вроде бегает, но как понять, что это мозг, а не случайность»
решается только сравнением с нулевой моделью. Скрипт гоняет одни и те же
сценарии на двух мозгах:

    настоящий     — ваш connectome.npz
    перемешанный  — та же исходящая степень и те же веса, но связи ведут
                    куда попало: физика та же, структуры нет

Если муха на настоящем коннектоме ведёт себя так же, как на перемешанном,
значит поведение даёт не коннектом, а кодировщик с декодером, и радоваться
нечему. Если отличается — отличие и есть вклад мозга, с размером эффекта.

    python3 scripts/behaviour_test.py --weight-scale 6
    python3 scripts/behaviour_test.py --synthetic 30000 --repeats 4
"""
import argparse
import math
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tests"))

from arena import Arena, EAST, NORTH, SOUTH, WEST      # noqa: E402
from flybrain.agent import BrainPool                    # noqa: E402
from flybrain.analysis import shuffle_connectome        # noqa: E402
from flybrain.config import Config                      # noqa: E402
from flybrain.connectome import Connectome, make_synthetic   # noqa: E402


# --------------------------------------------------------------------------
def _run(pool, arena, ticks, settle=25, mob="probe"):
    # Каждый повтор — своя муха: зерно агента выводится из его имени, и без
    # этого все повторы получаются побайтово одинаковыми, разброс нулевой,
    # а размер эффекта улетает в бесконечность.
    ag = pool.get(mob)
    for t in range(settle):                 # даём базовым линиям устояться
        ag.step(arena.percept(-1))
    mark = len(arena.path) - 1
    acts = []
    for t in range(ticks):
        out = ag.step(arena.percept(t))
        arena.apply(out)
        acts.append(out)
    pool.drop(mob)
    return acts, mark


# --- сценарии -------------------------------------------------------------
def scenario_phototaxis(pool, seed, ticks=90):
    """Тёмная комната, лампа сбоку. Мера: сближение с лампой."""
    a = Arena(seed=seed, ambient=0.06, clutter=10)
    a.lamp = (a.fx - 16, a.fy)
    _run(pool, a, ticks, mob=f"probe{seed}")
    return a.displacement_towards(a.lamp)


def scenario_escape(pool, seed, ticks=25):
    """Удар от соседа. Мера: удаление от обидчика после удара."""
    a = Arena(seed=seed, clutter=6)
    a.attacker = (a.fx + 2, a.fy)
    ag = pool.get(f"probe{seed}")
    for t in range(25):
        ag.step(a.percept(-1))
    for t in range(4):                      # пара спокойных тиков
        a.apply(ag.step(a.percept(t)))
    mark = len(a.path) - 1
    a.hit(0.9)
    for t in range(ticks):
        a.apply(ag.step(a.percept(100 + t)))
    pool.drop(f"probe{seed}")
    return -a.displacement_towards(a.attacker, since=mark)


def scenario_food(pool, seed, ticks=90):
    """Голодная муха, еда сбоку. Мера: сближение с едой плюс съеденное.

    Мерить одно только сближение нельзя: съеденная еда исчезает, и у самой
    успешной мухи метрика обнуляется. Поэтому цель запоминаем заранее и
    добавляем баллы за укусы.
    """
    a = Arena(seed=seed, clutter=10)
    a.nutrition = 90.0                       # почти голодает
    # В пределах обоняния: дальше 8 тайлов канал food_near равен нулю, и
    # сценарий мерил не пищевое поведение, а случайную прогулку.
    a.food = (a.fx, a.fy + 6)
    target = a.food
    _run(pool, a, ticks, mob=f"probe{seed}")
    return a.displacement_towards(target) + 3.0 * a.eaten


def scenario_bitter(pool, seed, ticks=40):
    """Горечь во рту. Мера: доля шагов назад."""
    a = Arena(seed=seed, clutter=6)
    a.bitter = 1.0
    acts, _ = _run(pool, a, ticks, mob=f"probe{seed}")
    back = sum(1 for x in acts if int(x.get("move") or 0) < 0)
    return back / max(1, len(acts))


def scenario_walls(pool, seed, ticks=120):
    """Тесная комната. Мера: сколько клеток обошла (больше — лучше)."""
    a = Arena(w=21, h=21, seed=seed, clutter=14)
    _run(pool, a, ticks, mob=f"probe{seed}")
    return len(set(a.path))


def scenario_booze(pool, seed, ticks=110):
    """Стакан со спиртом в стороне. Мера: сколько глотков сделала.

    Дрозофила летит на брожение — один из самых изученных её аппетитов.
    """
    a = Arena(seed=seed, clutter=8)
    a.booze = (a.fx + 5, a.fy)
    ag = pool.get(f"probe{seed}")
    for _ in range(30):
        ag.step(a.percept(-1))
    for t in range(ticks):
        a.apply(ag.step(a.percept(t)))
    pool.drop(f"probe{seed}")
    return a.drinks + 4.0 * a.drunk


def scenario_thirst(pool, seed, ticks=110):
    """Лужа в стороне, муха обезвожена. Мера: сближение с водой."""
    a = Arena(seed=seed, clutter=8)
    a.water = (a.fx, a.fy + 5)
    target = a.water
    ag = pool.get(f"probe{seed}")
    ag.encoder.thirst = 0.95          # канал ppk28 открыт только при жажде
    for _ in range(30):
        ag.step(a.percept(-1))
    ag.encoder.thirst = 0.95
    for t in range(ticks):
        a.apply(ag.step(a.percept(t)))
    pool.drop(f"probe{seed}")
    # Конечное смещение тут не годится: напившись, муха уходит, и у самой
    # успешной метрика обнуляется. Считаем сами глотки.
    return float(a.drinks)


def scenario_groom_site(pool, seed, ticks=30):
    """Тронули за конкретный бок. Мера: доля тиков, где чистит именно его.

    Щетинки у мухи сомато­топичны: раздражение места запускает чистку
    ЭТОГО места, а не абстрактный груминг.
    """
    a = Arena(seed=seed, clutter=4)
    ag = pool.get(f"probe{seed}")
    for _ in range(30):
        ag.step(a.percept(-1))
    hits = 0
    total = 0
    for t in range(ticks):
        if t % 6 == 0:
            # Трогаем за ЛЕВЫЙ БОК ТЕЛА, а не за левую сторону карты: муха
            # поворачивается, и в мировых координатах касание попадало бы
            # то в один бок, то в другой.
            a.touch_body(-1, 0)
        out = ag.step(a.percept(t))
        a.apply(out)
        if out.get("act") == "groom" and out.get("part"):
            total += 1
            if out["part"] == "левый бок":
                hits += 1
    pool.drop(f"probe{seed}")
    return hits / max(1, total) if total else 0.0


def scenario_climb(pool, seed, ticks=110):
    """Стол в стороне. Мера: залезла ли. После испуга муха ползёт вверх —
    на этом построен стандартный тест на локомоцию."""
    a = Arena(seed=seed, clutter=8)
    a.climbable = (a.fx - 4, a.fy + 2)
    ag = pool.get(f"probe{seed}")
    for _ in range(30):
        ag.step(a.percept(-1))
    for t in range(ticks):
        a.apply(ag.step(a.percept(t)))
    pool.drop(f"probe{seed}")
    return a.climbs + a.on_high


# Каждому сценарию — вид проверки:
#   "мозг"    — сравнение с перемешанным коннектомом имеет смысл: путь от
#               сенсорики до моторики должен идти ЧЕРЕЗ структуру связей;
#   "рефлекс" — дуга заложена руками (хемотаксис, вкус лапками, геотаксис),
#               и перемешанный мозг отработает её ровно так же. Сравнивать
#               с шумом бессмысленно, проверяется только "работает вообще".
SCENARIOS = [
    ("фототаксис: идёт ли на свет", scenario_phototaxis, "клеток к лампе", "мозг"),
    ("побег: уходит ли от обидчика", scenario_escape, "клеток от него", "мозг"),
    ("еда: идёт ли к еде голодной", scenario_food, "клеток к еде", "мозг"),
    ("горечь: пятится ли назад", scenario_bitter, "доля шагов назад", "мозг"),
    ("навигация: обход помещения", scenario_walls, "уникальных клеток", "мозг"),
    ("спирт: летит ли на брожение", scenario_booze, "глотков", "рефлекс"),
    ("жажда: идёт ли к воде", scenario_thirst, "глотков", "рефлекс"),
    ("чистка: то ли место чистит", scenario_groom_site, "доля попаданий", "рефлекс"),
    ("геотаксис: залезла ли наверх", scenario_climb, "подъёмов", "рефлекс"),
]

# Сколько должно получиться в рефлекторном сценарии, чтобы считать дугу
# работающей. Это не статистика, а функциональная проверка «сработало ли».
REFLEX_MIN = {
    "спирт: летит ли на брожение": 1.0,
    "жажда: идёт ли к воде": 1.0,
    "чистка: то ли место чистит": 0.6,
    "геотаксис: залезла ли наверх": 1.0,
}


# --------------------------------------------------------------------------
def cohen_d(a, b):
    """Размер эффекта для ПАРНЫХ замеров: d_z = среднее разностей / их СКО.

    Парность здесь не украшение, а главное. Раньше настоящий мозг получал
    арены с зёрнами 0..N, а перемешанный — 100..100+N, то есть сравнивались
    разные комнаты с разной расстановкой хлама и разной стартовой точкой.
    Разброс арены полностью маскировал разницу мозгов: на одном и том же
    коде один и тот же сценарий давал d от +1.07 до -0.14 от запуска к
    запуску. На одинаковых зёрнах разброс арены сокращается, и остаётся
    именно вклад коннектома.

    При нулевом разбросе делить нельзя — ставим пол, привязанный к масштабу
    самих величин, иначе d улетает в миллиарды.
    """
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return 0.0
    diff = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(diff)
    sd = math.sqrt(statistics.pvariance(diff))
    floor = 0.02 * (abs(statistics.fmean(a)) + abs(statistics.fmean(b))) + 1e-6
    return max(-99.0, min(99.0, m / max(sd, floor)))


def permutation_p(a, b, iters=20000, seed=0):
    """Парный перестановочный тест: случайно меняем знак каждой разности.

    Нулевая гипотеза — «какой из двух мозгов в этой комнате сработал лучше,
    решает монетка». Для парных данных это правильная перестановка;
    перемешивание всех замеров в общий котёл (как было) выбрасывает
    парность и занижает чувствительность.
    """
    if len(a) != len(b) or not a:
        return 1.0
    rng = np.random.default_rng(seed)
    diff = np.array([x - y for x, y in zip(a, b)], dtype=float)
    obs = abs(diff.mean())
    signs = rng.choice((-1.0, 1.0), size=(iters, len(diff)))
    stats = np.abs((signs * diff).mean(axis=1))
    return float((np.count_nonzero(stats >= obs - 1e-12) + 1) / (iters + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connectome")
    ap.add_argument("--config")
    ap.add_argument("--synthetic", type=int, nargs="?", const=30000)
    ap.add_argument("--weight-scale", type=float, dest="weight_scale")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--brain-ms", type=float, default=40.0)
    ap.add_argument("--only", help="подстрока названия сценария")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    print(f"конфиг: {cfg.loaded_from or 'не найден, всё по умолчанию'}")
    cfg.runtime.dt = a.dt
    cfg.runtime.brain_ms_per_tick = a.brain_ms
    cfg.runtime.adaptive = False
    if a.weight_scale is not None:
        cfg.runtime.weight_scale = a.weight_scale

    if a.synthetic:
        cx = make_synthetic(a.synthetic, 108, params=cfg.model)
        print(f"синтетический коннектом на {a.synthetic} нейронов")
        print("ВНИМАНИЕ: на синтетике этот тест почти ничего не значит.\n"
              "  Синтетический мозг — случайный слоистый граф, никакого\n"
              "  фототаксиса в нём не заложено, поэтому сравнивается\n"
              "  случайное со случайным: и 'РАБОТАЕТ', и 'не отличается'\n"
              "  одинаково ничего не доказывают. Тест имеет смысл только на\n"
              "  настоящей выгрузке FlyWire.")
    else:
        path = a.connectome or cfg.connectome_file
        cx = Connectome.load(path)
        print(f"коннектом: {path}")
    print(cx.describe())
    print(f"\nмножитель весов {cfg.runtime.weight_scale}, dt {cfg.runtime.dt} мс, "
          f"{cfg.runtime.brain_ms_per_tick} мс мозга на тик, повторов {a.repeats}")

    real = BrainPool(cfg, cx=cx)
    print("\nстрою перемешанный мозг (та же степень и веса, связи наугад)…",
          flush=True)
    null = BrainPool(cfg, cx=shuffle_connectome(cx, seed=7))

    rows = []
    reflex_rows = []
    for name, fn, unit, kind in SCENARIOS:
        if a.only and a.only.lower() not in name.lower():
            continue
        print(f"\n{name}…", flush=True)
        seeds = list(range(a.repeats))
        r = [fn(real, s) for s in seeds]
        if kind == "рефлекс":
            # Дуга заложена руками, перемешанный мозг отработает её так же.
            # Проверяем не «лучше ли шума», а «работает ли вообще».
            got = statistics.fmean(r)
            need = REFLEX_MIN.get(name, 1.0)
            hits = sum(1 for x in r if x >= need)
            reflex_rows.append((name, unit, got, need, hits, len(r)))
            print(f"  {got:.2f} {unit} (нужно {need:g}) — "
                  f"сработало в {hits} из {len(r)} комнат")
            continue
        # ОДНИ И ТЕ ЖЕ зёрна: та же комната, тот же хлам, та же стартовая
        # точка, та же муха — отличается только коннектом.
        z = [fn(null, s) for s in seeds]
        d = cohen_d(r, z)
        p = permutation_p(r, z, seed=1)
        # Знаковый счёт: в скольких комнатах настоящий мозг оказался лучше.
        # Метрика траектории шумная, и счёт побед по комнатам устойчивее
        # среднего — сходящиеся к одному и тому же выводу d и счёт дают
        # гораздо больше уверенности, чем любой из них по отдельности.
        wins_n = sum(1 for x, y in zip(r, z) if x > y)
        rows.append((name, unit, statistics.fmean(r), statistics.fmean(z),
                     d, p, wins_n, len(r)))
        print(f"  настоящий {statistics.fmean(r):+7.2f}   "
              f"перемешанный {statistics.fmean(z):+7.2f}   "
              f"d={d:+.2f}  p={p:.3f}  выиграл в {wins_n} из {len(r)} комнат")

    print("\n" + "=" * 86)
    print(f"{'сценарий':<32}{'мозг':>8}{'шум':>8}{'эффект':>8}{'p':>7}"
          f"{'комнат':>8}  вердикт")
    print("-" * 86)
    wins = 0
    for name, unit, mr, mz, d, p, wn, wtot in rows:
        if p < 0.05 and abs(d) > 0.8:
            verdict = "РАБОТАЕТ" if d > 0 else "работает НАОБОРОТ"
            wins += 1 if d > 0 else 0
        elif p < 0.2:
            verdict = "похоже на эффект, мало данных"
        else:
            verdict = "не отличается от шума"
        print(f"{name[:32]:<32}{mr:>8.2f}{mz:>8.2f}{d:>+8.2f}{p:>7.3f}"
              f"{wn:>5}/{wtot:<2}  {verdict}")

    print(f"\nСценариев, где настоящий коннектом объективно лучше: {wins} из {len(rows)}")

    if reflex_rows:
        print("\n" + "=" * 86)
        print("РЕФЛЕКТОРНЫЕ ДУГИ — заложены руками, с шумом не сравниваются")
        print("-" * 86)
        print(f"{'проверка':<34}{'получилось':>12}{'нужно':>8}{'комнат':>9}  вердикт")
        ok = 0
        for name, unit, got, need, hits, tot in reflex_rows:
            good = hits >= max(1, tot // 2)
            ok += 1 if good else 0
            print(f"{name[:34]:<34}{got:>12.2f}{need:>8.2g}{hits:>5}/{tot:<3}  "
                  f"{'работает' if good else 'НЕ СРАБОТАЛО'}")
        print(f"\nРефлексов работает: {ok} из {len(reflex_rows)}")
        print("Перемешанный мозг отработал бы их точно так же — это не"
              " вклад коннектома,")
        print("а заложенная дуга, и в dbg[\"source\"] она так и подписана.")
    print("\nd — парный размер эффекта Коэна (d_z): 0.8 считается большим.")
    print("Оба мозга гоняются на ОДНИХ И ТЕХ ЖЕ аренах, поэтому разброс")
    print("комнаты вычитается и остаётся вклад коннектома. p — парный")
    print("перестановочный тест со случайной сменой знаков. Малое число")
    print("повторов даёт большие p даже при настоящем эффекте: если интересно,")
    print("гоняйте --repeats 12 и больше, это единственный честный способ")
    print("отличить работающий мозг от красиво бегающего кодировщика.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
