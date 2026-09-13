"""Проверка рабочей точки при старте и, если надо, её подбор.

Зачем отдельный модуль. Настоящий коннектом при непрерывной сенсорной
накачке имеет ровно два вырожденных режима: при слабых весах он не спайкует
совсем, при сильных уходит в разгон и его давит гомеостаз. И тот и другой
снаружи выглядят одинаково — муха бегает, но нисходящие нейроны молчат, и
курс держит резервный контур. Поймать это по поведению невозможно, поэтому
сервер проверяет себя сам на старте.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

LADDER = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)


@dataclass
class Probe:
    weight_scale: float
    mean_hz: float
    inhib_mv: float
    dn_hz: float
    live_channels: int

    @property
    def silent(self) -> bool:
        """Мозг не спайкует вообще."""
        return self.mean_hz < 0.05

    @property
    def mute(self) -> bool:
        """Мозг живой, но ни одна команда не выходит."""
        return not self.silent and self.live_channels == 0

    @property
    def dead(self) -> bool:
        return self.silent or self.live_channels == 0

    @property
    def verdict(self) -> str:
        if self.runaway:
            return "разгон"
        if self.silent:
            return "сеть не спайкует"
        if self.mute:
            return "нисходящие молчат"
        return "норма"

    @property
    def runaway(self) -> bool:
        return self.mean_hz > 25.0 or self.inhib_mv > 40.0

    @property
    def ok(self) -> bool:
        return not self.dead and not self.runaway

    def score(self, target_hz: float) -> float:
        """Чем ближе к целевой частоте и чем больше живых каналов, тем лучше."""
        if not self.ok:
            return -1e9
        dist = abs(np.log((self.mean_hz + 1e-6) / max(target_hz, 1e-6)))
        return self.live_channels * 10.0 - dist * 3.0 - self.inhib_mv * 0.05


def probe_point(make_agent: Callable[[], "object"], ticks: int = 8,
                weight_scale: float = 1.0) -> Probe:
    """Прогнать несколько тиков с тонической засветкой и снять показатели."""
    from .decoders import COMMANDS
    ag = make_agent()
    percept = {"view_w": 15, "view_h": 15, "light": "5" * 225,
               "solid": "0" * 225, "mobs": "0" * 225, "items": "0" * 225}
    live = set()
    mean, inhib, dn = [], [], []
    for i in range(max(2, ticks)):
        ag.step(percept)
        res = ag.last_result
        if res is None:
            continue
        if i >= ticks // 2:              # первые тики — переходный процесс
            mean.append(float(res.mean_rate_hz))
            inhib.append(float(getattr(res, "inhib_mv", 0.0)))
            vals = [float(res.pop_rates.get(k, 0.0)) for k in COMMANDS
                    if k in res.pop_rates]
            dn.append(float(np.mean(vals)) if vals else 0.0)
            for k in COMMANDS:
                if float(res.pop_rates.get(k, 0.0)) > 0.0:
                    live.add(k)
    f = lambda a: float(np.mean(a)) if a else 0.0      # noqa: E731
    return Probe(weight_scale, f(mean), f(inhib), f(dn), len(live))


def autotune(make_agent_for: Callable[[float], "object"],
             current: float, target_hz: float = 2.0,
             ladder: Optional[List[float]] = None,
             ticks: int = 8,
             log: Callable[[str], None] = print) -> tuple[float, List[Probe]]:
    """Вернуть рабочий множитель весов и все снятые точки.

    Сначала проверяется текущее значение: если оно рабочее, ничего не
    меняется и лишнего времени не тратится. Перебор запускается, только если
    сеть мертва или в разгоне.
    """
    here = probe_point(lambda: make_agent_for(current), ticks, current)
    if here.ok:
        return current, [here]

    log(f"[flybrain] рабочая точка weight_scale={current:g} не годится: "
        f"{here.verdict} (мозг {here.mean_hz:.2f} Гц, тормоз "
        f"{here.inhib_mv:.0f} мВ, живых DN-каналов {here.live_channels}). "
        f"Подбираю.")
    seen = [here]
    for ws in (ladder or LADDER):
        if abs(ws - current) < 1e-9:
            continue
        p = probe_point(lambda: make_agent_for(ws), ticks, ws)
        seen.append(p)
        log(f"[flybrain]   x{ws:<5g} мозг {p.mean_hz:7.2f} Гц  "
            f"тормоз {p.inhib_mv:5.1f} мВ  DN {p.dn_hz:7.2f} Гц  "
            f"живых каналов {p.live_channels}")
    best = max(seen, key=lambda p: p.score(target_hz))
    if best.score(target_hz) < -1e8:
        log("[flybrain] рабочей точки в лестнице нет — модель поедет на "
            "резервном контуре. python3 scripts/explore.py --stats --paths")
        return current, seen
    return best.weight_scale, seen
