"""Один экземпляр мухи: коннектом -> LIF -> декодер -> действие в SS13."""
from __future__ import annotations

import collections
import threading
import zlib
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from .config import Config
from .connectome import Connectome
from .decoders import Action, MotorDecoder
from .encoders import Percept, SensoryEncoder
from .lif import LIFEngine, TickResult
from .populations import PopulationSet, hex_lattice, resolve


class BrainPool:
    """Разделяемая часть: коннектом и разбор популяций.

    Матрица связей read-only, поэтому все мухи на сервере пользуются одной
    копией. Индивидуальное у каждой мухи — только состояние мембран,
    это около 2 МБ на особь.
    """

    def __init__(self, cfg: Config, cx: Optional[Connectome] = None):
        self.cfg = cfg
        self.cx = cx if cx is not None else Connectome.load(cfg.connectome_file)
        self.pops: PopulationSet = resolve(self.cx, cfg.sensory, cfg.motor)
        self.hex = hex_lattice(15)
        self.agents: Dict[str, "FlyAgent"] = {}
        self._lock = threading.Lock()

    def get(self, mob_id: str) -> "FlyAgent":
        with self._lock:
            a = self.agents.get(mob_id)
            if a is None:
                # Зерно выводим из имени УСТОЙЧИВО. hash() для строк в питоне
                # рандомизируется при каждом запуске процесса, и один и тот
                # же прогон давал разную муху: воспроизвести баг по жалобе
                # было невозможно, а тесты плавали.
                a = FlyAgent(mob_id, self.cfg, self.cx, self.pops, self.hex,
                             seed=zlib.crc32(mob_id.encode("utf-8")) % (2 ** 31))
                self.agents[mob_id] = a
            return a

    def drop(self, mob_id: str) -> None:
        with self._lock:
            self.agents.pop(mob_id, None)

    def stats(self) -> dict:
        return {
            "neurons": self.cx.n,
            "connections": self.cx.nnz,
            "agents": list(self.agents.keys()),
            "populations": {
                "sensory": {k: len(v) for k, v in self.pops.sensory.items()},
                "motor": {k: len(v) for k, v in self.pops.motor.items()},
            },
            "source": self.cx.meta.get("source", "?"),
        }


@dataclass
class TickLog:
    tick: int
    wall_s: float
    brain_ms: float
    mean_hz: float
    state: str
    action: dict
    drive: Dict[str, float]
    dn_hz: Dict[str, float]


class FlyAgent:
    def __init__(self, mob_id: str, cfg: Config, cx: Connectome,
                 pops: PopulationSet, hexlat: np.ndarray, seed: int = 0):
        self.id = mob_id
        self.cfg = cfg
        self.pops = pops
        self.hex = hexlat
        self.engine = LIFEngine(cx, cfg.model, cfg.runtime, seed=seed)
        # Растр читается глазами, поэтому выборка не равномерная, а по слоям:
        # сверху сенсорика, снизу нисходящие, между ними остальной мозг.
        # При равномерной выборке из 139 тысяч нейронов вся видимая активность
        # схлопывается в тонкую полоску, и картинка ничего не сообщает.
        self.raster_bands = self._build_raster(cx, seed)
        self.engine.raster_idx = self.raster_bands["idx"]
        self.encoder = SensoryEncoder(pops, cfg.runtime)
        self.decoder = MotorDecoder()
        # Пишем не только моторику: зрительные популяции нужны оптомоторному
        # контуру, который подхватывает рулёжку, когда моторика молчит.
        self.record = dict(pops.motor_record())
        for name in ("vis_left", "vis_right"):
            pop = pops.sensory.get(name)
            if pop is not None and len(pop):
                self.record[name] = pop.idx
        self.brain_ms = cfg.runtime.brain_ms_per_tick
        self.ticks = 0
        self.created = time.time()
        self.history: Deque[TickLog] = collections.deque(maxlen=240)
        # Историю растра держим на сервере: так панель заполнена сразу при
        # открытии дашборда и переживает перезагрузку страницы.
        self.raster_hist: Deque[np.ndarray] = collections.deque(maxlen=160)
        self.last_percept: Dict[str, float] = {}
        self.last_action: Optional[Action] = None
        self.last_result: Optional[TickResult] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def _build_raster(self, cx, seed: int, total: int = 1024) -> dict:
        rng = np.random.default_rng(seed + 1)
        sens = np.concatenate([p.idx for p in self.pops.sensory.values()]) \
            if self.pops.sensory else np.empty(0, np.int64)
        motor = np.concatenate([p.idx for p in self.pops.motor.values()]) \
            if self.pops.motor else np.empty(0, np.int64)
        sens = np.unique(sens)
        motor = np.unique(motor)

        n_sens = min(len(sens), total // 4)
        n_motor = min(len(motor), total // 8)
        n_mid = max(0, total - n_sens - n_motor)

        pick_s = np.sort(rng.choice(sens, n_sens, replace=False)) if n_sens else sens[:0]
        pick_m = np.sort(rng.choice(motor, n_motor, replace=False)) if n_motor else motor[:0]
        used = np.union1d(pick_s, pick_m)
        pool = np.setdiff1d(np.arange(cx.n), used, assume_unique=False)
        n_mid = min(n_mid, len(pool))
        pick_c = np.sort(rng.choice(pool, n_mid, replace=False)) if n_mid else pool[:0]

        idx = np.concatenate([pick_s, pick_c, pick_m]).astype(np.int64)
        return {"idx": idx,
                "bands": [{"name": "сенсорика", "start": 0, "end": len(pick_s)},
                          {"name": "мозг", "start": len(pick_s),
                           "end": len(pick_s) + len(pick_c)},
                          {"name": "нисходящие",
                           "start": len(pick_s) + len(pick_c), "end": len(idx)}]}

    # ------------------------------------------------------------------
    def step(self, percept_json: dict) -> dict:
        with self._lock:
            t0 = time.perf_counter()
            p = Percept.from_json(percept_json)
            idx, hz = self.encoder.encode(p)
            # Сырые значения из игры держим отдельно: когда канал молчит,
            # первый вопрос — доходит ли вообще сигнал из DM или он теряется
            # уже в кодировщике. По дашборду это должно быть видно сразу.
            self.last_percept = {
                "здоровье": round(p.health, 2), "боль": round(p.pain, 2),
                "голод": round(p.hunger, 2), "касание": round(p.contact, 2),
                "еда рядом": round(p.food_near, 2),
                "еда в руках": round(p.food_touch, 2),
                "еда под ногами": round(p.food_underfoot, 2),
                "вода рядом": round(p.water_near, 2),
                "выпивка рядом": round(p.booze_near, 2),
                "спирт в крови": round(p.drunk, 2),
                "мебель рядом": round(p.climb_near, 2),
                "на возвышении": round(p.on_high, 2),
                "температура": round(p.temp_k - 273.15, 1),
                "давление": round(p.pressure_kpa, 1),
                "сытость": round(p.nutrition, 0) if p.nutrition >= 0 else "—",
                "удушье": round(p.suffocate, 2),
                "яд": round(p.toxin, 2), "горит": round(p.on_fire, 2),
                "оглушён": round(p.stunned, 2),
            }

            res = self.engine.run(self.brain_ms, idx, hz, record=self.record)
            self.last_result = res

            # Зрительные популяции читает оптомоторный контур, столкновение —
            # рефлекс на удар. Оба включаются, только если моторика молчит.
            sensory_rates = {k: v for k, v in res.pop_rates.items()
                             if k.startswith("vis_")}
            motor_rates = {k: v for k, v in res.pop_rates.items()
                           if not k.startswith("vis_")}
            action = self.decoder.decode(
                motor_rates, res.pop_spikes,
                sensory=sensory_rates, contact=p.contact,
                total_spikes=res.total_spikes,
                threat_bearing=p.threat_bearing(),
                food_bearing=p.food_bearing(),
                hunger=p.hunger,
                taste=max(p.food_touch, 0.8 * p.food_underfoot),
                food_near=p.food_near,
                thirst=self.encoder.thirst,
                water_near=p.water_near,
                water_bearing=p.water_bearing(),
                booze_near=p.booze_near,
                booze_bearing=p.booze_bearing(),
                drunk=self.encoder.effective_drunk(p.drunk),
                climb_near=p.climb_near,
                climb_bearing=p.climb_bearing(),
                on_high=p.on_high,
                touch_bearing=p.touch_bearing(),
                bitter=p.bitter)
            # Глоток снимает жажду. Держим здесь, потому что в кодовой базе
            # SS13 у человека гидратации обычно нет вовсе, и считать её
            # некому — а без этого муха пила бы не переставая.
            if action.act == "drink":
                self.encoder.drank()
            self.last_action = action
            self.ticks += 1

            # адаптация бюджета: если не укладываемся, муха живёт медленнее
            rt = self.cfg.runtime
            if rt.adaptive:
                if res.wall_s > rt.tick_budget_s and self.brain_ms > 5.0:
                    self.brain_ms = max(5.0, self.brain_ms * 0.8)
                elif (res.wall_s < rt.tick_budget_s * 0.5
                      and self.brain_ms < rt.brain_ms_per_tick):
                    self.brain_ms = min(rt.brain_ms_per_tick, self.brain_ms * 1.15)

            if res.raster is not None and len(res.raster):
                self.raster_hist.append(res.raster.astype(np.int16))

            self.history.append(TickLog(
                tick=self.ticks, wall_s=time.perf_counter() - t0,
                brain_ms=res.brain_ms, mean_hz=res.mean_rate_hz,
                state=action.dbg.get("state", ""), action=action.to_json(),
                drive=dict(self.encoder.last_drive),
                dn_hz={k: round(v, 2) for k, v in res.pop_rates.items()},
            ))
            out = action.to_json()
            out["brain"] = {
                "ms": round(res.brain_ms, 1),
                "wall_ms": round(res.wall_s * 1000, 1),
                "spikes": res.total_spikes,
                "mean_hz": round(res.mean_rate_hz, 3),
                "clipped": res.clipped,
                "source": action.source,
                "silent": self.decoder.brain_silent_for,
            }
            return out

    # ------------------------------------------------------------------
    def viz_frame(self, full: bool = False) -> dict:
        """Кадр для дашборда.

        full=True отдаёт всю накопленную историю растра (~100 КБ) — так
        панель заполнена сразу при открытии. Дальше идут только новые
        столбцы, иначе на 8 Гц набегает почти мегабайт в секунду.
        """
        with self._lock:
            hexvals = self.encoder.hex_projection(self.hex)
            res = self.last_result
            hist = list(self.history)[-120:]
            rast = list(self.raster_hist) if full else list(self.raster_hist)[-1:]
            return {
                "id": self.id,
                "tick": self.ticks,
                "hex": [round(float(v), 3) for v in hexvals],
                # компактно: для каждого кадра только индексы спайкнувших
                "raster": [np.flatnonzero(r).tolist() for r in rast],
                "raster_rows": len(self.raster_bands["idx"]),
                "raster_full": bool(full),
                "bands": self.raster_bands["bands"],
                "dn": {k: round(float(v), 2) for k, v in
                       (res.pop_rates.items() if res else [])},
                "z": self.decoder.snapshot(),
                "drive": {k: round(v, 3) for k, v in self.encoder.last_drive.items()},
                "percept": dict(self.last_percept),
                "action": self.last_action.to_json() if self.last_action else None,
                "brain": ({"spikes": res.total_spikes,
                           "mean_hz": round(res.mean_rate_hz, 3),
                           "wall_ms": round(res.wall_s * 1000, 1),
                           "ms": round(res.brain_ms, 1),
                           "inhib_mv": round(res.inhib_mv, 1),
                           "clipped": res.clipped} if res else {}),
                "brain_ms": round(self.brain_ms, 1),
                "mean_hz": [round(h.mean_hz, 3) for h in hist],
                "wall_ms": [round(h.wall_s * 1000, 1) for h in hist],
                "states": [h.state for h in hist],
                "image": ([round(float(v), 3) for v in self.encoder.last_image]
                          if self.encoder.last_image is not None else []),
                "view_w": self.encoder.w,
                "view_h": self.encoder.h,
                "diag": self.diagnose(),
                # Пороги нужны панели, чтобы рисовать засечки там же, где
                # они реально стоят, а не «примерно».
                # Внутренние состояния: жажда и толерантность живут в
                # кодировщике, тяга наверх и грязь — в декодере. Их не видно
                # ни в сенсорике, ни в моторике, а поведение они меняют.
                "inner": {
                    "жажда": round(self.encoder.thirst, 2),
                    "толерантность к спирту": round(self.encoder.tolerance, 2),
                    "тяга наверх": round(self.decoder.climb_urge, 2),
                    "цель": (self.last_action.dbg.get("goal")
                             if self.last_action else None) or "—",
                    "грязь": self.last_action.dbg.get("dirt", {})
                             if self.last_action else {},
                },
                "thresholds": {"act": self.decoder.act_threshold,
                               "escape": self.decoder.escape_threshold,
                               "move": self.decoder.move_threshold},
            }

    # ------------------------------------------------------------------
    def diagnose(self) -> dict:
        """Почему моторика молчит. Разбор по стадиям, а не общий совет.

        Три разные причины дают на дашборде одну и ту же надпись, а лечатся
        по-разному: популяция не нашлась в аннотациях; популяция есть, но
        мозг в целом не спайкует; мозг живой, а команды не поднимаются над
        своей базовой линией.
        """
        from .decoders import COMMANDS
        res = self.last_result
        rates = res.pop_rates if res else {}
        empty, quiet, active = [], [], []
        for k in COMMANDS:
            pop = self.pops.motor.get(k)
            if pop is None or len(pop) == 0:
                empty.append(k)
            elif float(rates.get(k, 0.0)) <= 1e-9:
                quiet.append(k)
            else:
                active.append(k)

        mean_hz = float(res.mean_rate_hz) if res else 0.0
        inhib = float(getattr(res, "inhib_mv", 0.0)) if res else 0.0
        ws = float(getattr(self.cfg.runtime, "weight_scale", 1.0))

        if len(empty) >= max(2, len(COMMANDS) // 2):
            cause, fix = "популяции не нашлись", (
                "Больше половины командных каналов пустые: дело не в усилении, "
                "а в разборе аннотаций. python3 scripts/explore.py --vocab "
                "покажет, какие типы вообще есть в вашей выгрузке.")
        elif mean_hz < 0.05:
            cause, fix = "мозг не спайкует", (
                f"Средняя частота {mean_hz:.3f} Гц при weight_scale = {ws:g}. "
                "python3 scripts/calibrate.py --sweep подберёт и СРАЗУ запишет "
                "рабочую точку в flybrain.json, дальше перезапустить сервер.")
        elif inhib > 40.0:
            cause, fix = "гомеостаз всё зажал", (
                f"Тормозной ток {inhib:.0f} мВ: сеть ушла в разгон и её "
                "прижало. Снизьте weight_scale или поднимите "
                "runtime.target_rate_hz.")
        elif not active:
            cause, fix = "команды не поднимаются над фоном", (
                "Мозг живой, но ни один DN-канал не выделяется. "
                "python3 scripts/calibrate.py покажет таблицу "
                "стимул -> отклик, python3 scripts/explore.py --inputs forward "
                "— кто на самом деле кормит команду.")
        else:
            cause, fix = "", ""

        return {"cause": cause, "fix": fix,
                "empty": empty, "quiet": quiet, "active": active,
                "mean_hz": round(mean_hz, 4), "inhib_mv": round(inhib, 1),
                "weight_scale": ws, "config": self.cfg.loaded_from or "",
                "silent_for": self.decoder.brain_silent_for}

    # ------------------------------------------------------------------
    def cloud(self, shape_n: int = 9000) -> dict:
        """Облако сомы нейронов для трёхмерной панели.

        Отдаётся один раз при открытии дашборда, дальше меняются только
        подсветки. Две части:

        * shape — случайная выборка по всему мозгу, тусклый силуэт. Это
          просто форма: где какие нейроны лежат.
        * live  — ровно те нейроны, за которыми мы следим растром. Их
          подсветка в каждом кадре берётся из того же массива спайков, что
          рисует растр, поэтому 3D-панель и растр показывают одно и то же,
          только одна в пространстве, а другая во времени.

        Ничего не додумывается: тусклые точки так и остаются тусклыми, а не
        «мигают для красоты» по средней частоте популяции.
        """
        cx = self.engine.cx
        xyz = np.asarray(cx.xyz, dtype=np.float64)
        live_idx = self.raster_bands["idx"]

        rng = np.random.default_rng(7)
        pool = np.arange(cx.n)
        shape_n = int(min(shape_n, cx.n))
        pick = np.sort(rng.choice(pool, shape_n, replace=False))

        if not np.any(xyz):
            # координат в выгрузке не было — раскладываем по сфере, чтобы
            # панель не врала пустотой, и честно говорим об этом
            ang = rng.random((cx.n, 2))
            th = ang[:, 0] * 2 * np.pi
            ph = np.arccos(2 * ang[:, 1] - 1)
            r = 0.6 + 0.4 * rng.random(cx.n)
            xyz = np.stack([r * np.sin(ph) * np.cos(th),
                            r * np.sin(ph) * np.sin(th),
                            r * np.cos(ph)], 1)
            real = False
        else:
            real = True

        c = np.median(xyz[xyz.any(axis=1)], axis=0) if real else xyz.mean(0)
        p = xyz - c
        # Масштаб берём по РАДИУСУ облака, а не по крайним координатам:
        # одна далёкая группа (или единичная кривая координата в выгрузке)
        # иначе сжимает весь остальной мозг в точку.
        rad = np.linalg.norm(p, axis=1)
        scale = float(np.percentile(rad[rad > 0], 95)) if np.any(rad > 0) else 1.0
        p = p / (scale or 1.0)
        p = np.clip(p, -2.5, 2.5)      # редкие выбросы не выносим за кадр
        # Оси упорядочиваем по размаху: самая широкая идёт по экрану
        # вправо, вторая вверх, самая узкая в глубину. Для мозга мухи это
        # и есть классический вид "спереди": две оптические доли по бокам.
        order = np.argsort(-p.std(0))
        p = p[:, order]

        bands = self.raster_bands["bands"]
        band_of = np.ones(len(live_idx), dtype=np.int8)
        for k, b in enumerate(bands):
            band_of[b["start"]:b["end"]] = k

        q = lambda a: [[round(float(v), 3) for v in row] for row in a]  # noqa: E731
        return {
            "shape": q(p[pick]),
            "live": q(p[live_idx]),
            "band": band_of.tolist(),
            "bands": [b["name"] for b in bands],
            "real_coords": real,
            "neurons": int(cx.n),
        }

    def hex_xy(self) -> List[List[float]]:
        return [[round(float(x), 4), round(float(y), 4)] for x, y in self.hex]
