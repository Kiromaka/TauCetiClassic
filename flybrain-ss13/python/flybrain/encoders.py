"""Кодирование того, что видит и чувствует персонаж в SS13, в нейронный вход.

На вход приходит пакет от DM-стороны (percept), на выходе — разреженный
вектор "индекс нейрона -> частота пуассоновской накачки в Гц".

Сетка поля зрения приходит уже в системе координат мухи: строка 0 — самая
дальняя вперёд, столбец 0 — крайний левый. Разворот по dir делает DM,
чтобы Python ничего не знал про BYOND-овые направления.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import RuntimeParams
from .populations import (Population, PopulationSet, build_view_map,
                          eye_sector)


# Пороги сытости SS13. Ноль голода при «сыт», единица при «голодает».
# Свежий человек рождается с NUTRITION_LEVEL_NORMAL = 330, то есть слегка
# голодным, — так и должно быть.
NUTRITION_WELL_FED = 440.0
NUTRITION_STARVING = 50.0


# --------------------------------------------------------------------------
@dataclass
class Percept:
    """Сенсорный пакет из игры."""
    tick: int = 0
    view_w: int = 15
    view_h: int = 15
    light: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    solid: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    mobs: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    items: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))

    health: float = 1.0        # 0..1
    pain: float = 0.0          # 0..1, дельта урона за тик
    hunger: float = 0.0        # 0..1
    temp_k: float = 293.0
    pressure_kpa: float = 101.0
    co2: float = 0.0           # 0..1
    toxin: float = 0.0         # 0..1, ядовитая атмосфера рядом
    food_near: float = 0.0     # 0..1, обратное расстояние до еды
    food_touch: float = 0.0    # 0..1, еда в руках / под ногами
    bitter: float = 0.0        # 0..1, невкусное во рту
    water_near: float = 0.0
    wind: float = 0.0          # 0..1, перепад давления
    sound: float = 0.0         # 0..1, громкость рядом
    contact: float = 0.0       # 0..1, столкновение / захват / толчок
    on_fire: float = 0.0
    stunned: float = 0.0
    suffocate: float = 0.0     # 0..1, кислородное голодание
    nutrition: float = -1.0    # сырое число сытости из игры, -1 = не прислано
    threat_fx: float = 0.0     # смещение ближайшего живого, вправо от мухи
    threat_fy: float = 0.0     # ... и вперёд
    threat_near: float = 0.0   # 0..1, насколько близко
    food_fx: float = 0.0       # смещение ближайшей еды, вправо от мухи
    food_fy: float = 0.0       # ... и вперёд
    # Спирт. Дрозофила летит на брожение — это не шутка, а отдельное
    # большое направление в науке: этанол привлекает на запах, малая доза
    # даёт гиперактивность, большая — потерю координации и седацию.
    booze_near: float = 0.0    # 0..1, обратное расстояние до выпивки
    booze_fx: float = 0.0
    booze_fy: float = 0.0
    drunk: float = 0.0         # 0..1, спирт в организме
    # Вода: канал ppk28. Жажду игра обычно не считает, поэтому мы держим
    # её сами; если DM пришлёт hydration, берём её.
    water_fx: float = 0.0
    water_fy: float = 0.0
    hydration: float = -1.0    # 0..1 из игры, -1 = не прислана
    # Мебель, на которую можно залезть: отрицательный геотаксис.
    climb_near: float = 0.0
    climb_fx: float = 0.0
    climb_fy: float = 0.0
    on_high: float = 0.0       # уже стоит на возвышении
    # Куда именно её тронули — щетинки у мухи сомато­топичны.
    touch_fx: float = 0.0
    touch_fy: float = 0.0
    # Тарзальный контакт: еда под ногами, на своей клетке. У мухи вкусовые
    # сенсиллы на лапках, и именно этот контакт запускает вытягивание
    # хоботка. Отдельно от food_touch (еда в руках = уже у рта).
    food_underfoot: float = 0.0

    @staticmethod
    def _bearing(fx: float, fy: float) -> Optional[float]:
        import math
        if fx == 0 and fy == 0:
            return None
        return math.degrees(math.atan2(fx, fy)) % 360.0

    def threat_bearing(self) -> Optional[float]:
        """Куда смотреть, чтобы видеть угрозу: градусы от курса мухи,
        по часовой стрелке. None, если рядом никого."""
        if self.threat_near <= 0:
            return None
        return self._bearing(self.threat_fx, self.threat_fy)

    def food_bearing(self) -> Optional[float]:
        """Куда идти к еде: градусы от курса мухи. None, если еды не видно."""
        if self.food_near <= 0:
            return None
        return self._bearing(self.food_fx, self.food_fy)

    def booze_bearing(self) -> Optional[float]:
        if self.booze_near <= 0:
            return None
        return self._bearing(self.booze_fx, self.booze_fy)

    def water_bearing(self) -> Optional[float]:
        if self.water_near <= 0:
            return None
        return self._bearing(self.water_fx, self.water_fy)

    def climb_bearing(self) -> Optional[float]:
        if self.climb_near <= 0:
            return None
        return self._bearing(self.climb_fx, self.climb_fy)

    def touch_bearing(self) -> Optional[float]:
        """С какой стороны тронули. None, если касания нет."""
        if self.contact <= 0.05:
            return None
        return self._bearing(self.touch_fx, self.touch_fy)

    @classmethod
    def from_json(cls, d: dict) -> "Percept":
        def arr(key, n):
            v = d.get(key)
            if v is None:
                return np.zeros(n, np.float32)
            if isinstance(v, str):
                # упакованный вид из DM: один символ '0'-'9' на клетку.
                # 225 чисел превращаются в 225 байт вместо полутора килобайт.
                a = (np.frombuffer(v.encode("ascii", "ignore"), dtype=np.uint8)
                     .astype(np.float32) - 48.0) / 9.0
                a = np.clip(a, 0.0, 1.0)
            else:
                a = np.asarray(v, dtype=np.float32).ravel()
            if a.size < n:
                a = np.pad(a, (0, n - a.size))
            return a[:n]

        w = int(d.get("view_w", 15))
        h = int(d.get("view_h", 15))
        n = w * h
        p = cls(tick=int(d.get("tick", 0)), view_w=w, view_h=h,
                light=arr("light", n), solid=arr("solid", n),
                mobs=arr("mobs", n), items=arr("items", n))
        for k in ("health", "pain", "hunger", "co2", "toxin", "food_near",
                  "food_touch", "food_underfoot",
                  "bitter", "water_near", "wind", "sound",
                  "contact", "on_fire", "stunned", "suffocate", "nutrition",
                  "threat_fx", "threat_fy", "threat_near",
                  "food_fx", "food_fy",
                  "booze_near", "booze_fx", "booze_fy", "drunk",
                  "water_fx", "water_fy", "hydration",
                  "climb_near", "climb_fx", "climb_fy", "on_high",
                  "touch_fx", "touch_fy"):
            if k in d:
                setattr(p, k, float(d[k]))
        # Голод пересчитываем из сырой сытости, если игра её прислала:
        # так пороги правятся здесь, без пересборки мира.
        if p.nutrition >= 0:
            p.hunger = float(np.clip(
                (NUTRITION_WELL_FED - p.nutrition)
                / max(1.0, NUTRITION_WELL_FED - NUTRITION_STARVING), 0.0, 1.0))
        if "temp" in d:
            p.temp_k = float(d["temp"])
        if "pressure" in d:
            p.pressure_kpa = float(d["pressure"])
        return p


# --------------------------------------------------------------------------
class SensoryEncoder:
    """Держит адаптационное состояние (муха адаптируется к средней яркости)."""

    # Дрозофиле комфортно около 25 C, но станция в SS13 живёт при 20 C,
    # и с точкой комфорта в 298 K муха постоянно «мёрзнет»: канал
    # thermo_cold висит на 0.4 весь раунд и забивает собой всё остальное.
    # Поэтому центр — станционные 20 C, плюс мёртвая зона, чтобы мелкие
    # колебания вообще не доходили до нейронов.
    T_COMFORT = 293.15
    T_DEADBAND = 4.0
    T_SPAN = 12.0

    def __init__(self, pops: PopulationSet, rt: Optional[RuntimeParams] = None,
                 view_w: int = 15, view_h: int = 15):
        self.pops = pops
        self.rt = rt or RuntimeParams()
        self.w, self.h = view_w, view_h
        self._adapt: Optional[np.ndarray] = None
        self._prev: Optional[np.ndarray] = None
        # След боли. Урон приходит из игры дельтой за тик: удар даёт всплеск
        # ровно на один тик, и на поведение это не успевает повлиять — ни
        # на дашборде не видно, ни мозг среагировать не успевает.
        # Ноцицепция у мухи фазная, но с последействием, поэтому держим
        # затухающий след примерно на секунду.
        self._pain_trace = 0.0
        self.pain_decay = 0.72
        # Жажда. В кодовой базе SS13 у человека обычно нет гидратации вовсе,
        # поэтому держим её сами: медленно растёт, сбрасывается питьём. Если
        # DM пришлёт hydration, берём её и свой счётчик не трогаем.
        self.thirst = 0.25
        self.thirst_rise = 1.0 / 900.0     # за тик; полный цикл ~3 минуты игры
        self.thirst_drink = 0.22           # сколько снимает один глоток
        # Толерантность к спирту. У мухи она развивается быстро и это
        # хорошо изученный эффект: та же доза со временем валит слабее.
        self.tolerance = 0.0
        self.tol_rise = 0.02
        self.tol_fall = 0.004
        # Доля неадаптирующейся яркости в зрительном ответе. Ноль — чистый
        # детектор контраста, муха света не замечает; слишком много — она
        # перестаёт различать движение на ярком фоне.
        self.phototaxis = 0.55
        # Постоянная Вебера: насколько тёмным должен быть фон, чтобы пятно
        # считалось ярким. Меньше — острее реакция на свет в темноте.
        self.weber_k = 0.15
        # Ширина функции чувствительности омматидия, в долях соседнего тайла.
        self.blur = 0.5
        # Доля обновления фонового уровня за тик. Медленная адаптация нужна,
        # чтобы неподвижная сцена не исчезала из виду целиком.
        self.tau_adapt = 0.12

        # карта нейрон -> ячейка поля зрения, с учётом полусфер
        self.view_map: Dict[str, np.ndarray] = {}
        self.eye_cells: Dict[str, np.ndarray] = {}
        for name in ("vis_left", "vis_right"):
            pop = pops.sensory.get(name)
            if pop is None or len(pop) == 0:
                continue
            proj = getattr(self.rt, "projection", "perspective")
            # Сектор глаза задаётся азимутом: левый глаз смотрит влево и
            # немного за нос, правый зеркально. Поля перекрываются спереди
            # и сзади — вдвоём глаза закрывают почти полный круг.
            a_from, a_to = eye_sector(
                name, getattr(self.rt, "eye_fov_deg", 195.0),
                getattr(self.rt, "eye_overlap_deg", 25.0))
            self.view_map[name] = build_view_map(
                pop, view_w, view_h, projection=proj,
                eye_height=getattr(self.rt, "eye_height_tiles", 0.6),
                d_near=getattr(self.rt, "near_tiles", 0.75),
                azim_from_deg=a_from, azim_to_deg=a_to)
            # При перспективе сектор уже задан азимутом, отдельная маска по
            # столбцам сетки только выбрасывала бы половину нейронов.
            self.eye_cells[name] = (np.ones(view_w * view_h, dtype=bool)
                                    if proj != "grid" else self._hemifield(name))

        # последний посчитанный образ — для визуализации
        self._hex_map: Optional[np.ndarray] = None
        self.last_image: Optional[np.ndarray] = None
        self.last_drive: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _hemifield(self, name: str) -> np.ndarray:
        """Маска ячеек, попадающих в поле зрения одного глаза.

        У дрозофилы поля зрения глаз перекрываются спереди примерно на 20 гр.,
        поэтому центральная полоса достаётся обоим.
        """
        gx = np.arange(self.w)
        if name == "vis_left":
            m = gx <= int(self.w * 0.6)
        else:
            m = gx >= int(self.w * 0.4) - 1
        return np.tile(m, self.h)

    # ------------------------------------------------------------------
    def _blur(self, cells: np.ndarray) -> np.ndarray:
        """Размытие по сетке поля: сепарабельное ядро [b, 1, b], нормированное.

        Физический смысл — функция чувствительности омматидия. У дрозофилы
        она шире межомматидиального угла, поэтому точечный источник
        засвечивает несколько соседних фасеток, а не ровно одну.
        """
        b = float(self.blur)
        if b <= 0:
            return cells
        g = cells.reshape(self.h, self.w)
        out = g.copy()
        out[:, :-1] += b * g[:, 1:]
        out[:, 1:] += b * g[:, :-1]
        norm = np.ones_like(g)
        norm[:, :-1] += b
        norm[:, 1:] += b
        out /= norm
        g2 = out
        out = g2.copy()
        norm = np.ones_like(g2)
        out[:-1, :] += b * g2[1:, :]
        out[1:, :] += b * g2[:-1, :]
        norm[:-1, :] += b
        norm[1:, :] += b
        out /= norm
        return out.reshape(-1)

    # ------------------------------------------------------------------
    def image(self, p: Percept) -> np.ndarray:
        """Собрать "яркость" каждой клетки так, как её увидел бы глаз мухи.

        Стены — тёмные пятна, живые объекты и предметы — контрастные пятна.
        Муха реагирует прежде всего на контраст, а не на абсолютный уровень.
        """
        n = self.w * self.h
        light = p.light if p.light.size == n else np.zeros(n, np.float32)
        solid = p.solid if p.solid.size == n else np.zeros(n, np.float32)
        mobs = p.mobs if p.mobs.size == n else np.zeros(n, np.float32)
        items = p.items if p.items.size == n else np.zeros(n, np.float32)

        img = np.clip(light, 0, 1)
        img = img * (1.0 - 0.85 * np.clip(solid, 0, 1))     # стены гасят свет
        img = np.clip(img + 0.55 * mobs + 0.25 * items, 0, 1.5)
        return img.astype(np.float32)

    # ------------------------------------------------------------------
    def visual_drive(self, p: Percept) -> Tuple[np.ndarray, np.ndarray]:
        """Контраст и временная производная по клеткам поля зрения."""
        img = self.image(p)
        if self._adapt is None or self._adapt.shape != img.shape:
            self._adapt = img.copy()
            self._prev = img.copy()
        contrast = img - self._adapt
        motion = img - self._prev
        self._prev = img
        self._adapt += self.tau_adapt * (img - self._adapt)
        self.last_image = img
        return contrast, motion

    # ------------------------------------------------------------------
    def encode(self, p: Percept) -> Tuple[np.ndarray, np.ndarray]:
        """Вернуть (индексы нейронов, частоты в Гц)."""
        fmax = self.rt.sensory_max_hz
        idx_parts: List[np.ndarray] = []
        hz_parts: List[np.ndarray] = []
        drive_log: Dict[str, float] = {}

        if p.hydration < 0:
            self.thirst = float(np.clip(self.thirst + self.thirst_rise, 0.0, 1.0))

        # ---- зрение ---------------------------------------------------
        contrast, motion = self.visual_drive(p)
        # L1 — ON-канал, L2 — OFF; здесь берём обе полярности разом:
        # сила ответа = |контраст| + подчёркнутое движение + доля яркости.
        #
        # Яркость нужна отдельным слагаемым, иначе фототаксиса не получается:
        # лампа в тёмной комнате неподвижна, контраст к ней адаптируется за
        # десяток тиков, и муха перестаёт её "видеть". Дрозофила же на свет
        # идёт, и идёт устойчиво. Поэтому часть ответа не адаптируется.
        img = self.last_image if self.last_image is not None else np.zeros_like(contrast)
        # Яркость входит НЕ абсолютным уровнем, а контрастом по Веберу
        # относительно средней яркости поля. Ламинарные нейроны мухи делают
        # ровно это — делят на среднюю освещённость. Разница принципиальная:
        # при абсолютном уровне равномерно освещённая комната ровно светит
        # во все стороны и никуда не тянет, зато лампа в темноте почти не
        # выделяется на фоне тонического уровня. По Веберу наоборот — лампа
        # в темноте даёт максимум, равномерный свет не даёт ничего.
        mean_l = float(img.mean())
        weber = np.clip((img - mean_l) / (mean_l + self.weber_k), 0.0, 1.0)
        cell_drive = np.clip(np.abs(contrast) + 1.6 * np.abs(motion)
                             + self.phototaxis * weber, 0, 1.0)
        # Оптика глаза размывает: поле зрения одного омматидия шире, чем шаг
        # между соседями. Без размытия яркая точка в пяти тайлах попадает в
        # одну-две фасетки из четырёхсот и тонет в тоническом уровне.
        cell_drive = self._blur(cell_drive)
        for name in ("vis_left", "vis_right"):
            pop = self.pops.sensory.get(name)
            if pop is None or len(pop) == 0:
                continue
            vm = self.view_map[name]
            eye = self.eye_cells[name]
            d = cell_drive[vm] * eye[vm]
            # Тонический уровень: ламинарные нейроны у мухи активны и на
            # неподвижной сцене, контраст лишь модулирует этот уровень.
            # Без тоники зрительный тракт молчит и муха стоит столбом.
            d = np.clip(0.32 + 0.68 * d, 0, 1)
            idx_parts.append(pop.idx)
            hz_parts.append(d * fmax)
            drive_log[name] = float(d.mean())

        # ---- прочие каналы -------------------------------------------
        def add(pop_name: str, level: float, label: Optional[str] = None):
            pop = self.pops.sensory.get(pop_name)
            level = float(np.clip(level, 0.0, 1.0))
            drive_log[label or pop_name] = level
            if pop is None or len(pop) == 0 or level <= 0:
                return
            idx_parts.append(pop.idx)
            hz_parts.append(np.full(len(pop), level * fmax, dtype=np.float64))

        # Голод. Интероцептивный канал, если в снапшоте нашлись подходящие
        # нейроны; если нет, голод всё равно поднимает усиление аппетитивных
        # путей — это документированный эффект, а не выдумка.
        add("hunger", p.hunger)
        hungry_pop = self.pops.sensory.get("hunger")
        appetitive = p.food_near
        if hungry_pop is None or len(hungry_pop) == 0:
            appetitive = max(appetitive, 0.45 * p.hunger)

        # обоняние: еда притягивает, огонь/яд/трупы отталкивают
        # Запах спирта идёт по тому же аппетитивному пути, что и еда:
        # для дрозофилы брожение это и есть еда, дрожжи в гнилом фрукте.
        # Привлекательность падает, когда муха уже пьяна.
        booze_pull = p.booze_near * (1.0 - 0.6 * min(1.0, p.drunk))
        add("olf_attractive",
            0.9 * max(appetitive, 0.85 * booze_pull) * (0.35 + 0.65 * p.hunger))
        add("olf_ethanol", booze_pull)
        add("olf_aversive", max(p.toxin, p.on_fire))
        add("olf_co2", max(p.co2, 0.8 * p.suffocate))

        # вкус
        # Вкус сахара. У мухи вкусовые сенсиллы не только на хоботке, но и
        # НА ЛАПКАХ: наступила на еду — уже попробовала, и именно тарзальный
        # контакт запускает вытягивание хоботка. Раньше канал молчал, пока
        # еда не окажется в руках, то есть путь "нашла — попробовала — съела"
        # начинался с середины и не запускался никогда.
        taste = max(float(p.food_touch), 0.8 * float(p.food_underfoot))
        add("gust_sugar", taste * (0.4 + 0.6 * p.hunger))
        add("gust_bitter", p.bitter)
        # Вода. Канал ppk28 у мухи работает как детектор воды, и открывается
        # он только при обезвоживании — сытая водой муха на лужу не реагирует.
        if p.hydration >= 0:
            self.thirst = float(np.clip(1.0 - p.hydration, 0.0, 1.0))
        add("gust_water", p.water_near * (0.2 + 0.8 * self.thirst), "gust_water")
        add("thirst", self.thirst)

        # механика: ветер/звук в джонстонов орган, касание в щетинки
        add("wind_jo", max(p.wind, 0.7 * p.sound))
        # Касание — только касание. Раньше сюда же подмешивалась боль, и
        # канал висел на 0.7 от постоянных столкновений, полностью пряча
        # редкие всплески урона.
        add("bristle", np.clip(p.contact, 0, 1))
        add("mechano_any", 0.5 * p.stunned)

        # боль отдельным каналом, со следом
        self._pain_trace = max(float(p.pain), self._pain_trace * self.pain_decay)
        if self._pain_trace < 1e-3:
            self._pain_trace = 0.0
        add("nocic", np.clip(1.3 * self._pain_trace + 0.6 * p.on_fire, 0, 1))

        # температура
        dev = p.temp_k - self.T_COMFORT
        dev = 0.0 if abs(dev) <= self.T_DEADBAND else (
            dev - np.sign(dev) * self.T_DEADBAND)
        dt_hot = dev / self.T_SPAN
        add("thermo_hot", max(0.0, dt_hot) + 0.8 * p.on_fire)
        add("thermo_cold", max(0.0, -dt_hot))

        # Толерантность копится от присутствия спирта в организме и медленно
        # спадает. Считаем здесь, потому что это состояние тела, а не решение.
        if p.drunk > 0.05:
            self.tolerance = min(1.0, self.tolerance + self.tol_rise * p.drunk)
        else:
            self.tolerance = max(0.0, self.tolerance - self.tol_fall)
        # В drive_log не кладём: это не сенсорный канал, а состояние тела,
        # и на дашборде оно живёт в панели внутренних состояний.

        self.last_drive = drive_log
        if not idx_parts:
            return np.empty(0, np.int64), np.empty(0, np.float64)
        idx = np.concatenate(idx_parts).astype(np.int64)
        hz = np.concatenate(hz_parts).astype(np.float64)
        # один нейрон мог попасть в две популяции — берём максимум
        uniq_idx, inv = np.unique(idx, return_inverse=True)
        hz_max = np.zeros(len(uniq_idx))
        np.maximum.at(hz_max, inv, hz)
        return uniq_idx, hz_max

    # ------------------------------------------------------------------
    def drank(self, amount: float = 1.0) -> None:
        """Муха сделала глоток: жажда падает."""
        self.thirst = float(np.clip(self.thirst - self.thirst_drink * amount,
                                    0.0, 1.0))

    def effective_drunk(self, drunk: float) -> float:
        """Опьянение с поправкой на толерантность.

        У дрозофилы толерантность развивается быстро, и это один из самых
        воспроизводимых эффектов в этой области: та же доза со временем
        валит слабее.
        """
        return float(np.clip(drunk * (1.0 - 0.55 * self.tolerance), 0.0, 1.0))

    # ------------------------------------------------------------------
    def hex_projection(self, hex_xy: np.ndarray) -> np.ndarray:
        """Спроецировать последний образ на решётку омматидиев для дашборда.

        Проекция та же, что у нейронов, иначе картинка врёт: на ней клетки
        шли бы один в один, а сеть видела бы перспективу.
        """
        if self.last_image is None:
            return np.zeros(len(hex_xy), np.float32)
        if self._hex_map is None or len(self._hex_map) != len(hex_xy):
            fake = Population("hex", np.arange(len(hex_xy)), hex_xy)
            # Панель показывает объединённое поле обоих глаз, поэтому
            # азимут берём полным кругом.
            self._hex_map = build_view_map(
                fake, self.w, self.h,
                projection=getattr(self.rt, "projection", "perspective"),
                eye_height=getattr(self.rt, "eye_height_tiles", 0.6),
                d_near=getattr(self.rt, "near_tiles", 0.75),
                azim_from_deg=-180.0, azim_to_deg=180.0)
        return self.last_image[self._hex_map]
