#!/usr/bin/env python3
"""Калибровка: какие сенсорные популяции до каких нисходящих нейронов доходят.

Это главный инструмент для настройки на настоящем коннектоме. Скрипт по
очереди накачивает каждую сенсорную популяцию (как оптогенетикой в статье
Shiu et al.) и печатает, какие DN на это отвечают и насколько сильно
относительно фона.

    python3 scripts/calibrate.py
    python3 scripts/calibrate.py --gain 0.5 --settle 40 --probe 30

Если строка стимула вся нулевая, значит путь до моторики в модели не
пробивается: крутите gain, input_drive_mv или homeostasis.target_rate_hz.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flybrain.config import Config                      # noqa: E402
from flybrain.connectome import Connectome, make_synthetic  # noqa: E402
from flybrain.lif import LIFEngine                      # noqa: E402
from flybrain.populations import resolve                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome", default=None)
    ap.add_argument("--config")
    ap.add_argument("--synthetic", type=int, nargs="?", const=30000)
    ap.add_argument("--gain", type=float)
    ap.add_argument("--drive", type=float, help="input_drive_mv")
    ap.add_argument("--settle", type=int, default=25, help="тиков на успокоение")
    ap.add_argument("--probe", type=int, default=15, help="тиков на замер")
    ap.add_argument("--tick-ms", type=float, default=40.0)
    ap.add_argument("--level", type=float, default=0.9, help="сила стимула 0..1")
    ap.add_argument("--weight-scale", type=float, dest="weight_scale",
                    help="рантайм-множитель весов (не требует пересборки .npz)")
    ap.add_argument("--raw", action="store_true",
                    help="не вычитать общую моду (по умолчанию вычитается, "
                         "потому что декодер тоже её вычитает)")
    ap.add_argument("--sweep", action="store_true",
                    help="искать рабочую точку: перебрать множитель весов")
    ap.add_argument("--sweep-values", default="0.5,1,1.5,2,3,4,6,8,12",
                    help="через запятую, какие множители пробовать")
    ap.add_argument("--apply", action="store_true",
                    help="сразу записать найденную рабочую точку в flybrain.json")
    ap.add_argument("--no-apply", action="store_true",
                    help="только напечатать, ничего не записывать")
    a = ap.parse_args()

    cfg = Config.load(a.config)
    print(f"конфиг: {cfg.loaded_from or 'не найден, всё по умолчанию'}")
    if a.gain is not None:
        cfg.model.gain = a.gain
    if a.drive is not None:
        cfg.runtime.input_drive_mv = a.drive
    if a.weight_scale is not None:
        cfg.runtime.weight_scale = a.weight_scale
    cfg.runtime.dt = max(cfg.runtime.dt, 1.0)

    if a.synthetic:
        cx = make_synthetic(a.synthetic, 108, params=cfg.model)
        print(f"синтетический коннектом на {a.synthetic} нейронов")
    else:
        path = a.connectome or cfg.connectome_file
        cx = Connectome.load(path)
        print(f"коннектом: {path}")
    print(cx.describe())

    pops = resolve(cx, cfg.sensory, cfg.motor)
    print()
    print(pops.summary())

    motor = {k: v.idx for k, v in pops.motor.items() if len(v)}
    sensory = [(k, v.idx) for k, v in pops.sensory.items() if len(v)]
    fmax = cfg.runtime.sensory_max_hz

    def measure(stim_idx, stim_hz, settle=None, probe=None):
        settle = a.settle if settle is None else settle
        probe = a.probe if probe is None else probe
        eng = LIFEngine(cx, cfg.model, cfg.runtime, seed=1)
        for _ in range(settle):
            eng.run(a.tick_ms, stim_idx, stim_hz, motor)
        acc = {k: 0.0 for k in motor}
        spikes = {k: 0 for k in motor}
        mean = 0.0
        for _ in range(probe):
            r = eng.run(a.tick_ms, stim_idx, stim_hz, motor)
            for k in motor:
                acc[k] += r.pop_rates[k]
                spikes[k] += r.pop_spikes[k]
            mean += r.mean_rate_hz
        return ({k: v / probe for k, v in acc.items()},
                mean / probe, float(eng.inhib), spikes)

    vis = np.concatenate([pops.sensory_idx("vis_left"), pops.sensory_idx("vis_right")])
    names = [k for k in motor if k not in ("dn_all", "dn_left", "dn_right")]

    if a.sweep:
        return sweep(cx, cfg, pops, motor, vis, sensory, names, measure, a, fmax)

    print("\nфон (только тоника зрения)…")
    base, base_mean, base_inh, base_spk = measure(vis, np.full(len(vis), fmax * 0.32))

    w = max(12, max((len(n) for n in names), default=12))
    mode = ("без вычитания общей моды" if a.raw
            else "с вычитанием общей моды — так же, как её вычитает декодер")
    print(f"\nотклики нисходящих нейронов, {mode}")
    print(f"\n{'стимул':<18}{'мозг Гц':>9}{'торм':>6} | "
          + "".join(f"{n[:9]:>10}" for n in names))
    print("-" * (33 + 10 * len(names)))
    print(f"{'фон':<18}{base_mean:>9.2f}{base_inh:>6.1f} | "
          + "".join(f"{base[n]:>10.1f}" for n in names))
    any_response = False
    responses = {}

    for label, idx in sensory:
        if label.startswith("vis_"):
            stim = vis.copy()
            hz = np.full(len(vis), fmax * 0.32)
            mask = np.isin(vis, idx)
            hz[mask] = fmax * a.level
        else:
            stim = np.concatenate([vis, idx])
            hz = np.concatenate([np.full(len(vis), fmax * 0.32),
                                 np.full(len(idx), fmax * a.level)])
        rates, mean, inh, spk = measure(stim, hz)
        raw = [rates[n] - base[n] for n in names]
        # Декодер вычитает общую моду: подъём, одинаковый для всех каналов,
        # он сигналом не считает. Таблица показывает то же самое, иначе
        # самый шумный канал выглядит так, будто реагирует на всё подряд.
        common = 0.0 if a.raw else float(np.median(raw))
        deltas = []
        for n, d0 in zip(names, raw):
            d = d0 - common
            if abs(d) >= 1.0:
                any_response = True
            responses.setdefault(n, {})[label] = d
            deltas.append(f"{'.':>10}" if abs(d) < 1.0 else f"{d:>+10.1f}")
        print(f"{label:<18}{mean:>9.2f}{inh:>6.1f} | " + "".join(deltas))

    # --- сходится ли проводка ----------------------------------------
    # Абсолютный порог "отклик больше герца" тут не годится: у канала из
    # трёх нейронов частота квантована огромными ступенями, и он "отвечает"
    # на что угодно, а у крупной популяции настоящий отклик может быть
    # скромным. Считаем сильным отклик от 40% собственного максимума канала.
    print()
    print("проводка: отвечает ли канал на то, на что должен")
    print("-" * 78)
    want = {}
    for sname, spec in cfg.sensory.items():
        for target in getattr(spec, "affinity", []) or []:
            want.setdefault(target, []).append(sname)

    for n in names:
        vals = responses.get(n, {})
        pop = len(pops.motor.get(n, []))
        if not vals:
            print(f"  {n:<12}{pop:>5} нейронов  молчит")
            continue
        peak = max(abs(v) for v in vals.values())
        strong = [k for k, v in vals.items()
                  if abs(v) >= max(2.0, 0.4 * peak)]
        expect = want.get(n, [])
        hit = [e for e in expect if e in strong]
        marks = []
        if pop and pop <= 3:
            marks.append("популяция из трёх нейронов — частота квантована, "
                         "каналу верить нельзя")
        if peak < 1.0:
            marks.append("до канала не докручивается")
        elif len(strong) >= max(4, int(0.55 * len(sensory))):
            marks.append(f"сильный отклик на {len(strong)} стимулов из "
                         f"{len(sensory)} — похоже на индикатор общей активности")
        if expect:
            if hit:
                marks.append("ожидаемые входы на месте: " + ", ".join(hit))
            else:
                marks.append("ОЖИДАЕМЫХ ВХОДОВ НЕТ (" + ", ".join(expect)
                             + ") — тянут другие")
        print(f"  {n:<12}{pop:>5} нейронов  пик {peak:>7.1f} Гц  "
              f"сильных откликов {len(strong)}")
        for m in marks:
            print(f"               {m}")
        if strong:
            top = sorted(vals.items(), key=lambda kv: -abs(kv[1]))[:4]
            print("               тянут сильнее всех: "
                  + ", ".join(f"{k} {v:+.0f}" for k, v in top))

    tiny = [n for n in names if 0 < len(pops.motor.get(n, [])) <= 3]
    if tiny:
        print()
        print(f"Каналы {', '.join(tiny)} собраны из трёх нейронов и меньше.")
        print("Такая популяция насыщается от любого входа: в игре это"
              " выглядит как")
        print("работающий мозг, но информации о стимуле в ней нет. Поискать"
              " настоящую")
        print("популяцию в этих же данных умеет поиск по графу:")
        for n in tiny[:2]:
            src = {"feed": "gust_sugar", "escape": "nocic",
                   "backward": "gust_bitter"}.get(n, "nocic")
            print(f"    python3 scripts/find_channel.py --for {n} --from {src}")

    print("\nЧитается так: цифра — прирост частоты DN относительно фона, в Гц.")
    print("Точка значит, что канал до этой команды не докрутился.")
    if not any_response:
        print("\nВНИМАНИЕ: ни один стимул не сдвинул ни одну команду.")
        if sum(base.values()) <= 0.01:
            print("Нисходящие нейроны молчат и в фоне — сеть работает слишком тихо.")
        else:
            print("Нисходящие нейроны активны, но одинаково — различий нет.")
        print("Порядок действий:")
        print("  1. python3 scripts/explore.py --stats --paths")
        print("     проверить, есть ли путь до моторики вообще и не выбрана ли")
        print("     сетчаткой популяция без исходящих синапсов")
        print("  2. python3 scripts/calibrate.py --sweep")
        print("     подобрать множитель весов, при котором DN оживают")
    return 0


# --------------------------------------------------------------------------
def sweep(cx, cfg, pops, motor, vis, sensory, names, measure, a, fmax) -> int:
    """Найти рабочую точку: при каком множителе весов оживают DN.

    Слишком тихо — нисходящие молчат и таблица пустая. Слишком громко —
    всё горит одинаково и различий нет. Между этими двумя режимами обычно
    есть окно в полтора-два порядка по множителю, его и ищем.
    """
    values = [float(v) for v in a.sweep_values.split(",") if v.strip()]
    # два стимула-зонда: зрение слева и самый связный несенсорный канал
    probes = [("vis_left", pops.sensory_idx("vis_left"))]
    for cand in ("gust_sugar", "olf_attractive", "bristle", "wind_jo"):
        idx = pops.sensory_idx(cand)
        if len(idx):
            probes.append((cand, idx))
            break

    print(f"\nсвип множителя весов: {values}")
    print(f"зонды: {', '.join(n for n, _ in probes)}\n")
    print(f"{'x весов':>8}{'мозг Гц':>10}{'торм мВ':>9}{'DN фон Гц':>11}"
          f"{'ответили':>10}{'макс |dHz|':>12}   вердикт")
    print("-" * 78)

    best = None
    for ws in values:
        cfg.runtime.weight_scale = ws
        base, base_mean, base_inh, _ = measure(vis, np.full(len(vis), fmax * 0.32),
                                               settle=max(8, a.settle // 2),
                                               probe=max(6, a.probe))
        dn_base = sum(base.values()) / max(1, len(base))
        responded, biggest = 0, 0.0
        for label, idx in probes:
            stim = vis.copy()
            hz = np.full(len(vis), fmax * 0.32)
            mask = np.isin(vis, idx)
            if mask.any():
                hz[mask] = fmax * a.level
            else:
                stim = np.concatenate([vis, idx])
                hz = np.concatenate([hz, np.full(len(idx), fmax * a.level)])
            rates, mean, inh, _ = measure(stim, hz,
                                          settle=max(8, a.settle // 2),
                                          probe=max(6, a.probe))
            for n in names:
                d = abs(rates[n] - base[n])
                if d > max(1.0, 0.15 * max(base[n], 1.0)):
                    responded += 1
                biggest = max(biggest, d)

        # Вердикт по трём признакам: жива ли сеть, не насыщены ли DN,
        # и есть ли РАЗЛИЧИЯ между стимулами. Различия важнее абсолютных
        # частот: решение принимается по z-оценке относительно базовой линии.
        silent = dn_base < 0.05 and biggest < 1.0
        runaway = base_mean > 25 or base_inh > 15
        saturated = dn_base > 150

        if base_mean < 0.05 and base_inh > 5:
            verdict = "разгон, задавлен торможением"
        elif base_mean < 0.05:
            verdict = "мертво"
        elif silent:
            verdict = "DN молчат"
        elif runaway:
            verdict = "разгон"
        elif responded == 0 and saturated:
            verdict = "насыщение"
        elif responded == 0:
            verdict = "нет различий"
        else:
            verdict = "РАБОЧАЯ ТОЧКА"
            # чем больше откликнувшихся каналов, тем лучше; за сильное
            # торможение и за насыщение штрафуем, при равенстве берём
            # меньший множитель — там меньше риск уехать в разгон
            score = (responded
                     - 0.08 * base_inh
                     - (1.5 if saturated else 0.0)
                     - 0.02 * ws)
            if best is None or score > best[1]:
                best = (ws, score, base_mean, responded)
        print(f"{ws:>8.2f}{base_mean:>10.2f}{base_inh:>9.1f}{dn_base:>11.2f}"
              f"{responded:>10}{biggest:>12.1f}   {verdict}")

    print()
    if best:
        print(f"Рекомендую weight_scale = {best[0]:g} "
              f"(средняя частота {best[2]:.2f} Гц, откликнулось {best[3]} DN-каналов).")
        if a.no_apply:
            print("Прописать в flybrain.json вручную:")
            print(f'    "runtime": {{ "weight_scale": {best[0]:g} }}')
        else:
            # По умолчанию записываем сами. Печатать "пропишите в конфиг" —
            # ровно тот шаг, на котором настройка и терялась: свип отработал,
            # сервер перезапустили, а файл остался прежним.
            where = cfg.patch_file({"runtime": {"weight_scale": float(best[0])}})
            print(f"Записал weight_scale = {best[0]:g} в {where}")
            print("Теперь перезапустите сервер — он подхватит файл сам:")
            print("    python3 -m flybrain.server")
        print("Проверить полной таблицей:")
        print(f"    python3 scripts/calibrate.py --weight-scale {best[0]:g}")
    else:
        print("Рабочей точки в этом диапазоне нет. Что делать дальше:")
        print("  python3 scripts/explore.py --stats --paths")
        print("Если у сенсорных популяций почти нет исходящих связей или в")
        print("таблице путей стоят точки — дело не в усилении, а в точке входа:")
        print("  python3 scripts/explore.py --inputs forward")
        print("покажет, кто на самом деле кормит эту команду.")
        print("Можно также расширить диапазон: --sweep-values 10,20,40,80")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
