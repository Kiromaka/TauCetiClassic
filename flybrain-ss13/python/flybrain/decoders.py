"""Декодирование активности нисходящих нейронов в действия персонажа SS13.

Абсолютные частоты у разных популяций DN отличаются на порядки, а после
смены коннектома/усиления вообще уплывают. Поэтому решение принимается не
по абсолютному порогу, а по z-оценке относительно скользящей базовой линии
самой популяции: "этот DN сейчас активнее, чем обычно".

Приоритетная лестница поведения повторяет то, как это устроено у мухи:
эскейп перебивает всё, потом угроза, потом задний ход, груминг, еда,
и только потом обычная ходьба.
"""
from __future__ import annotations

import collections
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Каналы-команды, по которым работает приоритетная лестница.
COMMANDS = ("escape", "forward", "backward", "turn_left", "turn_right",
            "groom", "threat", "feed", "speed")
# Служебные популяции: в лестнице не участвуют, нужны для рулёжки и
# для вычитания общей моды.
AUX = ("dn_left", "dn_right", "dn_all")

# BYOND-овые направления
NORTH, SOUTH, EAST, WEST = 1, 2, 4, 8
DIR_BY_ANGLE = [NORTH, EAST, SOUTH, WEST]        # 0, 90, 180, 270 градусов


@dataclass
class Action:
    seq: int = 0
    dir: int = NORTH
    move: int = 0            # 1 вперёд, -1 назад, 0 стоять
    run: int = 0             # 1 бежать, 0 шагом
    act: Optional[str] = None    # groom | eat | drink | climb | threat | escape | halt
    part: Optional[str] = None   # для груминга: какую часть чистит
    say: Optional[str] = None
    source: str = "мозг"         # кто на самом деле принял это решение
    dbg: Dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {"seq": self.seq, "dir": self.dir, "move": self.move,
             "run": self.run, "act": self.act}
        if self.part:
            d["part"] = self.part
        if self.say:
            d["say"] = self.say
        d["dbg"] = self.dbg
        return d


@dataclass
class Goal:
    """Одна аппетитивная мотивация: дойти до чего-то и что-то с ним сделать.

    Еда, вода, выпивка и «залезть повыше» отличаются только силой тяги,
    пеленгом и тем, что делать по прибытии, — поэтому это один механизм, а
    не четыре особых случая в лестнице.
    """
    name: str
    drive: float                      # 0..1 с учётом приоритета
    bearing: Optional[float]
    arrived: bool
    act: str                          # что уходит в игру по прибытии
    go_state: str                     # подпись состояния по дороге
    busy_state: str                   # подпись состояния на месте
    source: str                       # чем помечать решение в dbg
    hold: int = 4                     # тиков между порциями действия


Z_CLAMP = 12.0


def poisson_var(rate_hz: float, spikes: float) -> float:
    """Нижняя граница дисперсии оценки частоты, из счёта спайков.

    Частота считается как spikes / (N нейронов * T секунд). Счёт спайков
    пуассоновский, поэтому var(частоты) >= частота / (N*T), а N*T = spikes /
    частота, откуда var >= частота^2 / spikes. Ниже этого разброс физически
    опуститься не может: это шум самого измерения.

    Зачем это нужно. Экспоненциальная оценка дисперсии на ровном канале
    сходится к его собственному микрошуму: на 34.7 Гц со случайным разбросом
    0.3 Гц СКО падает до 0.3, и скачок на 5 Гц (14% от среднего!) даёт
    z = 16.6. Такой канал НАВСЕГДА залипает наверху приоритетной лестницы,
    и муха до конца раунда пятится назад или чистится, что бы вокруг ни
    происходило. Пуассоновский пол это чинит: тот же скачок даёт z ~ 1.5.
    """
    if spikes <= 0.0:
        return 0.01
    return max(rate_hz * rate_hz / spikes, 0.01)


class Baseline:
    """Экспоненциальная база (среднее и разброс) частоты популяции."""

    def __init__(self, halflife_ticks: float = 40.0):
        self.a = 1.0 - 0.5 ** (1.0 / max(1.0, halflife_ticks))
        self.mean = 0.0
        self.var = 1.0
        self.n = 0
        self.floor = 0.01          # физический пол дисперсии, см. poisson_var

    def update(self, x: float, floor: Optional[float] = None) -> float:
        if floor is not None:
            self.floor = max(floor, 0.01)
        self.n += 1
        if self.n <= 2:
            self.mean = x
            self.var = max(1.0, x)
            return 0.0
        d = x - self.mean
        self.mean += self.a * d
        self.var = (1 - self.a) * (self.var + self.a * d * d)
        return self._z(d)

    def _z(self, d: float) -> float:
        sd = math.sqrt(max(self.var, self.floor, 1e-6))
        return max(-Z_CLAMP, min(Z_CLAMP, d / sd))

    def z(self, x: float, floor: Optional[float] = None) -> float:
        if floor is not None:
            self.floor = max(floor, 0.01)
        if self.n <= 2:
            return 0.0
        return self._z(x - self.mean)


class MotorDecoder:
    def __init__(self,
                 turn_gain: float = 34.0,     # градусов на единицу z за тик
                 move_threshold: float = 0.35,
                 act_threshold: float = 1.8,
                 escape_threshold: float = 2.4,
                 heading_snap: float = 45.0,
                 halflife_ticks: float = 80.0,
                 spontaneous: bool = True,
                 arousal_gain: float = 1.2,
                 arousal_rise: float = 0.06,
                 arousal_drain: float = 0.25,
                 optomotor_gain: float = 0.9,
                 fallback: bool = True,
                 common_mode: bool = True,
                 pop_steering: float = 1.0,
                 turn_clamp: float = 1.1,
                 dir_hysteresis: float = 14.0,
                 dir_min_ticks: int = 2,
                 noise_tau: float = 0.25,
                 evidence_ticks: int = 3,
                 turn_tau: float = 0.22,
                 bias_tau: float = 0.06,
                 turn_deadband: float = 0.12,
                 appetite_gain: float = 1.1,
                 feed_hunger: float = 0.25,
                 feed_ticks: int = 4,
                 thirst_threshold: float = 0.35,
                 chemo_gain: float = 0.8):
        self.turn_gain = turn_gain
        self.move_threshold = move_threshold
        self.act_threshold = act_threshold
        self.escape_threshold = escape_threshold
        self.heading_snap = heading_snap
        self.spontaneous = spontaneous
        # Спонтанная локомоторная тяга. У мухи ходьба идёт вспышками, и
        # интервалы между ними задаются внутренним состоянием, а не стимулом.
        # Переменная накапливается, пока муха стоит, и тратится на ходу —
        # получается та самая структура "постоял, побежал, постоял".
        self.arousal_gain = arousal_gain
        self.arousal_rise = arousal_rise
        self.arousal_drain = arousal_drain
        self.arousal = 0.0
        # Резервный контур. Включается, только когда нисходящие нейроны не
        # дают вообще никакого сигнала. Это НЕ мозг мухи и за него не
        # выдаётся: источник текущего решения всегда виден в dbg["source"]
        # и на дашборде. Без него муха с молчащим мозгом упирается в первую
        # же стену и толкает её до конца раунда, а понять по поведению, что
        # мозг не работает, невозможно.
        self.optomotor_gain = optomotor_gain
        self.fallback = fallback
        self.source = "мозг"
        self.brain_silent_for = 0
        self.stuck_for = 0
        self._flip = 1
        self._flip_left = 2                # сколько ударов подряд крутим в ту же сторону
        # Вычитание общей моды. Без него любой стимул, поднимающий общий
        # уровень активности мозга, протекает во ВСЕ каналы разом, и самый
        # чувствительный из них выглядит так, будто он реагирует на всё.
        self.common_mode = common_mode
        # Вклад популяционной асимметрии нисходящих в рулёжку.
        self.pop_steering = pop_steering
        # Пищевой контур: сила аппетитивного поиска, порог голода для
        # рефлекса хоботка и выдержка между попытками поесть.
        self.appetite_gain = appetite_gain
        self.feed_hunger = feed_hunger
        self.feed_ticks = feed_ticks
        self.thirst_threshold = thirst_threshold
        # Тяга залезть повыше: копится в покое, подскакивает от испуга.
        self.climb_urge = 0.0
        self._rng = random.Random(1234)
        # Куда её тронули и насколько давно: след затухает, поэтому
        # муха чистится ещё несколько секунд после касания.
        self._dirt: Dict[str, float] = {}
        self.dirt_decay = 0.82
        self.chemo_gain = chemo_gain
        self._act_hold = 0
        self.turn_clamp = turn_clamp
        self.dir_hysteresis = dir_hysteresis
        self.dir_min_ticks = dir_min_ticks
        self.noise_tau = noise_tau
        # Сколько тиков спайков накапливать перед решением.
        self.evidence_ticks = evidence_ticks
        self._ev: Dict[str, collections.deque] = {}
        self._ev_rate: Dict[str, float] = {}
        self.turn_tau = turn_tau
        self.bias_tau = bias_tau
        self.turn_deadband = turn_deadband
        self._turn_lp = 0.0
        self._turn_bias = 0.0
        self._noise_asym = 0.0
        self._dir_angle = 0.0              # текущая сторона света, градусы
        self._dir_hold = 0                 # тиков до следующей смены dir
        self._opto_base = Baseline(24.0)   # адаптация к постоянному перекосу
        self._saccade_in = 0               # тиков прямого хода до поворота
        self.base: Dict[str, Baseline] = {}
        self.halflife = halflife_ticks
        self.heading = 0.0          # градусы, 0 = север
        self.seq = 0
        self.state = "idle"
        self._refract = 0           # тики "занятости" после эскейпа и т.п.
        self._last_z: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _z(self, name: str, rates: Dict[str, float]) -> float:
        if name not in rates:
            return 0.0
        b = self.base.setdefault(name, Baseline(self.halflife))
        return b.update(float(rates[name]))

    # ------------------------------------------------------------------
    def _bearing_error(self, bearing: float) -> float:
        """Насколько довернуть курс, чтобы смотреть на цель. Градусы, ±180.

        Игра меряет пеленг от СТОРОНЫ СВЕТА, на которую повёрнут спрайт
        (dir), а курс внутри декодера непрерывный, и эти две величины
        расходятся до 45 градусов плюс гистерезис. Если подмешивать пеленг
        прямо в курс, муха промахивается мимо цели и наматывает вокруг неё
        круги — ровно то, что было видно с едой.
        """
        absolute = (self._dir_angle + bearing) % 360.0
        return ((absolute - self.heading + 180.0) % 360.0) - 180.0

    # Иерархия груминга у дрозофилы идёт спереди назад: раздражение
    # головы подавляет чистку брюшка, но не наоборот. Порядок — приоритет.
    GROOM_ORDER = ("голова", "левый бок", "правый бок", "брюшко")

    def _groom_site(self, touch_bearing: Optional[float], contact: float,
                    bitter: float) -> Optional[str]:
        """Какую часть тела чистить.

        Щетинки у мухи сомато­топичны: раздражение конкретного места
        запускает чистку ИМЕННО ЭТОГО места, а не абстрактный груминг. Плюс
        горечь во рту — классический триггер чистки головы и хоботка.
        """
        if bitter > 0.25:
            self._dirt["голова"] = max(self._dirt.get("голова", 0.0), bitter)
        if contact > 0.25 and touch_bearing is not None:
            b = touch_bearing % 360.0
            if b < 45 or b >= 315:
                site = "голова"
            elif b < 135:
                site = "правый бок"
            elif b < 225:
                site = "брюшко"
            else:
                site = "левый бок"
            self._dirt[site] = min(1.0, self._dirt.get(site, 0.0) + contact)
        # след затухает: муха чистится ещё несколько секунд после касания
        for k in list(self._dirt):
            self._dirt[k] *= self.dirt_decay
            if self._dirt[k] < 0.05:
                del self._dirt[k]
        for site in self.GROOM_ORDER:
            if self._dirt.get(site, 0.0) > 0.12:
                return site
        return None

    # ------------------------------------------------------------------
    def _pick_goal(self, z, hunger, taste, food_near, food_bearing,
                   thirst, water_near, water_bearing,
                   booze_near, booze_bearing, drunk,
                   climb_near, climb_bearing, on_high) -> Optional["Goal"]:
        """Самая сильная сейчас мотивация, либо None.

        Веса приоритетов не выдуманы: обезвоживание убивает быстрее голода,
        поэтому жажда перебивает еду; брожение для дрозофилы это тоже еда,
        но чуть менее ценная, чем сахар; «залезть повыше» — фоновая тяга,
        которая уступает всему остальному, зато подскакивает после испуга.
        """
        goals: List[Goal] = []

        # --- еда --------------------------------------------------------
        brain_feed = z.get("feed", 0.0) >= self.act_threshold
        if food_near > 0 or brain_feed or taste > 0.25:
            drive = self.appetite_gain * hunger * food_near
            eat_now = brain_feed or (self.fallback and taste > 0.25
                                     and hunger > self.feed_hunger)
            goals.append(Goal(
                name="еда", drive=max(drive, 1.0 if eat_now else 0.0),
                bearing=food_bearing, arrived=eat_now, act="eat",
                go_state="к еде", busy_state="ест",
                source="мозг" if brain_feed else "вкус+голод",
                hold=self.feed_ticks))

        # --- вода: канал ppk28 открыт только при обезвоживании ----------
        if water_near > 0 and thirst > self.thirst_threshold:
            drive = 1.15 * thirst * water_near
            goals.append(Goal(
                name="вода", drive=drive, bearing=water_bearing,
                arrived=water_near >= 0.99, act="drink",
                go_state="к воде", busy_state="пьёт",
                source="жажда", hold=self.feed_ticks))

        # --- брожение: этанол привлекает, пьяную — заметно меньше --------
        if booze_near > 0:
            appetite = 0.35 + 0.65 * hunger
            drive = 0.9 * appetite * booze_near * (1.0 - 0.7 * min(1.0, drunk))
            goals.append(Goal(
                name="спирт", drive=drive, bearing=booze_bearing,
                arrived=booze_near >= 0.87, act="drink",
                go_state="к выпивке", busy_state="пьёт спирт",
                source="брожение", hold=self.feed_ticks + 2))

        # --- отрицательный геотаксис ------------------------------------
        # Муха упорно ползёт вверх, и после испуга особенно — на этом
        # построен стандартный тест на локомоцию (RING assay).
        if climb_near > 0 and on_high < 0.5:
            # Тяга ВВЕРХ от расстояния не зависит: у мухи отрицательный
            # геотаксис это не «хочу вон тот стол», а «хочу выше», и
            # расстояние только выбирает, на что лезть. Когда тягу множили
            # на обратное расстояние, муха видела стол в шести клетках,
            # получала 0.05 и проходила мимо.
            drive = 0.40 * (self.climb_urge + 0.15)
            goals.append(Goal(
                name="верх", drive=drive, bearing=climb_bearing,
                arrived=climb_near >= 0.87, act="climb",
                go_state="наверх", busy_state="забирается",
                source="геотаксис", hold=6))

        if not goals:
            return None
        best = max(goals, key=lambda g: g.drive)
        return best if (best.drive > 0.15 or best.arrived) else None

    # ------------------------------------------------------------------
    def _feeding(self, z: Dict[str, float], hunger: float, taste: float) -> bool:
        """Кормиться ли сейчас.

        Основной путь — пищевые нисходящие нейроны (MN9 и компания). Но в
        части выгрузок FlyWire эта популяция не находится вовсе: канал
        мёртвый, z всегда ноль, и муха не ест НИКОГДА, сколько еды под ней
        ни лежи. Поэтому есть второй путь — тарзальный вкус плюс голод: у
        настоящей мухи это и есть рефлекс вытягивания хоботка, дуга
        "сахарный рецептор на лапке -> хоботок", и она работает даже у
        обезглавленной мухи. Источник решения виден в dbg["source"].
        """
        if z.get("feed", 0.0) >= self.act_threshold:
            return True
        return bool(self.fallback and taste > 0.25 and hunger > self.feed_hunger)

    # ------------------------------------------------------------------
    def decode(self, rates: Dict[str, float],
               spikes: Optional[Dict[str, int]] = None,
               sensory: Optional[Dict[str, float]] = None,
               contact: float = 0.0,
               total_spikes: int = 0,
               threat_bearing: Optional[float] = None,
               food_bearing: Optional[float] = None,
               hunger: float = 0.0,
               taste: float = 0.0,
               food_near: float = 0.0,
               thirst: float = 0.0,
               water_near: float = 0.0,
               water_bearing: Optional[float] = None,
               booze_near: float = 0.0,
               booze_bearing: Optional[float] = None,
               drunk: float = 0.0,
               climb_near: float = 0.0,
               climb_bearing: Optional[float] = None,
               on_high: float = 0.0,
               touch_bearing: Optional[float] = None,
               bitter: float = 0.0) -> Action:
        """rates — Гц на нейрон по каждой моторной популяции за прошедший тик.

        sensory — частоты сенсорных популяций (их читает оптомоторный контур),
        contact — сигнал столкновения из игры, total_spikes — активность по
        всему мозгу, она же источник собственного шума сети.

        food_bearing/hunger/taste/food_near — пищевой контур: куда пахнет
        едой, насколько муха голодна и есть ли вкусовой контакт (лапками или
        хоботком). Всё это ИДЁТ В МОЗГ как сенсорика, а сюда попадает ещё
        раз только затем, чтобы было чем рулить, когда пищевые нисходящие
        нейроны в выгрузке не нашлись. Источник решения всегда виден в
        dbg["source"].
        """
        self.seq += 1
        # Молчание считаем по каналам-командам: служебные dn_left/dn_right
        # могут шевелиться, но если ни одна команда не отвечает, за муху
        # мозг всё равно не решает.
        if sum(abs(rates.get(k, 0.0)) for k in COMMANDS) <= 1e-9:
            self.brain_silent_for += 1
        else:
            self.brain_silent_for = 0
        # z-оценки считаем ДО обновления баз, чтобы можно было вычесть
        # общую моду, и только потом обновляем базы сырыми частотами.
        keys = COMMANDS + AUX
        sp = spikes or {}
        # Окно накопления. Частота за ОДИН тик по популяции из четырёх
        # нейронов — это пять-шесть спайков, то есть пуассоновский шум
        # измерения около 12 Гц. На настоящем коннектоме пользователя
        # ноцицепция поднимала backward на +12.5 Гц, то есть ровно на шум:
        # z выходил 1.0 при пороге 1.26, и муха на удар не реагировала.
        # Считая спайки за три тика, шум падает в корень из трёх, и тот же
        # отклик даёт z около 1.8. Это то же самое измерение, просто за
        # 150 мс мозгового времени вместо 50.
        win = max(1, int(self.evidence_ticks))
        z_raw = {}
        for k in keys:
            b = self.base.setdefault(k, Baseline(self.halflife))
            if k not in rates:
                z_raw[k] = 0.0
                continue
            hist = self._ev.setdefault(k, collections.deque(maxlen=win))
            if hist.maxlen != win:                    # окно поменяли на ходу
                hist = collections.deque(hist, maxlen=win)
                self._ev[k] = hist
            hist.append((float(rates[k]), float(sp.get(k, 0.0))))
            r = sum(x for x, _ in hist) / len(hist)   # средняя частота за окно
            s_tot = sum(y for _, y in hist)           # спайков за всё окно
            self._ev_rate[k] = r
            # var(средней частоты) = частота / (N*T*окно) = частота^2 / спайки
            z_raw[k] = b.z(r, floor=poisson_var(r, s_tot))

        common = 0.0
        if self.common_mode:
            vals = sorted(z_raw[k] for k in COMMANDS if k in rates)
            if len(vals) >= 3:
                m = len(vals) // 2
                common = vals[m] if len(vals) % 2 else 0.5 * (vals[m - 1] + vals[m])

        z = {k: z_raw[k] - (common if k in COMMANDS else 0.0) for k in keys}
        for k in keys:
            if k in rates:
                hist = self._ev[k]
                r = sum(x for x, _ in hist) / len(hist)
                self.base[k].update(r, floor=poisson_var(
                    r, sum(y for _, y in hist)))

        self._common = common
        self._last_z = {k: z[k] for k in COMMANDS}

        act: Optional[str] = None
        move = 0
        run = 0
        ladder_source: Optional[str] = None

        if self._refract > 0:
            self._refract -= 1
        if self._act_hold > 0:
            self._act_hold -= 1

        groom_site = self._groom_site(touch_bearing, contact, bitter)
        part: Optional[str] = None
        goal = self._pick_goal(
            z=z, hunger=hunger, taste=taste, food_near=food_near,
            food_bearing=food_bearing,
            thirst=thirst, water_near=water_near, water_bearing=water_bearing,
            booze_near=booze_near, booze_bearing=booze_bearing, drunk=drunk,
            climb_near=climb_near, climb_bearing=climb_bearing, on_high=on_high)
        aim: Optional[float] = None

        # ---- приоритетная лестница -----------------------------------
        if z["escape"] >= self.escape_threshold:
            # Гигантский нейрон: рывок ПРОЧЬ ОТ УГРОЗЫ. Раньше тут был
            # разворот на 180 градусов от текущего курса — муха убегала
            # "назад", а не от того, кто её ударил, и с равной вероятностью
            # бежала обидчику навстречу.
            if threat_bearing is not None:
                # Пеленг приходит от стороны света, на которую повёрнут
                # спрайт, а не от непрерывного курса — иначе побег уводит
                # мимо обидчика на угол между ними.
                self.heading = (self._dir_angle + threat_bearing + 180.0) % 360.0
            else:
                self.heading += 180.0
            act, move, run = "escape", 1, 1
            self.state = "escape"
            self._refract = 3

        elif self._refract > 0:
            act, move, run = None, 1, 1
            self.state = "escape-run"

        elif z["threat"] >= self.act_threshold:
            act, self.state = "threat", "threat"

        elif z["backward"] >= self.act_threshold * 0.7:
            move, self.state = -1, "backward"

        elif z["groom"] >= self.act_threshold or groom_site is not None:
            act, self.state = "groom", "groom"
            part = groom_site
            if part:
                self.state = "чистит " + part
                self._dirt[part] = max(0.0, self._dirt[part] - 0.34)
                if z["groom"] < self.act_threshold:
                    ladder_source = "щетинки"

        elif goal is not None and goal.arrived:
            # Дошла: остаётся на месте и делает своё дело порциями. Муха на
            # капле еды или воды не уходит после первого касания хоботком —
            # без этого аппетитивный поиск тут же снимал её с клетки, она
            # возвращалась, и так по кругу, ничего не съев.
            self.state = goal.busy_state
            if self._act_hold <= 0:
                act = goal.act
                self._act_hold = goal.hold
            ladder_source = goal.source
            aim = goal.bearing

        else:
            fwd = z["forward"] + self.arousal_gain * self.arousal
            if goal is not None:
                # Аппетитивный поиск: муха идёт к источнику. Это отдельная от
                # "просто ходьбы" мотивация, и без неё внутреннее состояние
                # вообще ни на что не влияет — голод и жажда видны на
                # дашборде, а поведение от них не меняется.
                move = 1
                run = 1 if goal.drive > 0.55 else 0
                self.state = goal.go_state
                aim = goal.bearing
                if z["forward"] < self.move_threshold:
                    ladder_source = goal.source
            elif fwd >= self.move_threshold:
                move = 1
                run = 1 if (fwd + z["speed"]) >= self.act_threshold else 0
                self.state = "forward"
            else:
                self.state = "idle"

        # ---- тяга залезть повыше ---------------------------------------
        # Копится в покое, подскакивает от боли и угрозы, тратится на
        # подъёме. После испуга муха ползёт вверх — это и есть основа
        # стандартного теста на локомоцию.
        startled = max(z.get("escape", 0.0), z.get("threat", 0.0)) > 1.0
        if on_high > 0.5 or act == "climb":
            self.climb_urge = max(0.0, self.climb_urge - 0.25)
        elif startled:
            self.climb_urge = min(1.0, self.climb_urge + 0.35)
        else:
            self.climb_urge = min(1.0, self.climb_urge + 0.01)

        # ---- обновление спонтанной тяги --------------------------------
        if move != 0 or act is not None:
            self.arousal = max(0.0, self.arousal - self.arousal_drain)
        else:
            self.arousal = min(1.0, self.arousal + self.arousal_rise)

        # ---- рулёжка --------------------------------------------------
        turn_drive = z["turn_right"] - z["turn_left"]
        # К именованным рулевым нейронам добавляем асимметрию всей нисходящей
        # популяции: на трёх нейронах сигнал квантован и шумен, на сотнях —
        # устойчив. Общая мода из обоих уже вычтена, так что остаётся
        # именно перекос лево/право, а не общий подъём активности.
        if self.pop_steering and "dn_left" in rates and "dn_right" in rates:
            turn_drive += self.pop_steering * (z["dn_right"] - z["dn_left"])
        self.source = "мозг"
        have_brain = abs(turn_drive) > 1e-3

        # Снятие ПОСТОЯННОГО перекоса. Разность z двух рулевых каналов не
        # нулевая в среднем: частоты популяций считаются из горстки спайков
        # за тик, распределение счёта скошено вправо, и у более "взрывного"
        # канала большинство тиков оказывается ниже собственного среднего.
        # Получается стабильная добавка (на синтетике -0.53), которая
        # упирает рулёжку в ограничитель и кругами водит муху до конца
        # раунда — это и есть главная причина верчения вокруг своей оси.
        #
        # Физиологически рулит не абсолютный перекос лево/право, а его
        # ИЗМЕНЕНИЕ: постоянная разница означала бы, что муха всю жизнь
        # хромает в одну сторону. Поэтому медленно вычитаем среднее.
        # Первые тики усредняем как обычное среднее, дальше экспоненциально:
        # иначе свежая муха первые полминуты ездит с неснятым перекосом.
        # Постоянная подобрана под окно накопления: частоты в окне
        # скоррелированы между тиками, и слишком медленный трекер за ними
        # не успевает — остаточный дрейф был 5.8 гр/тик вместо 1.5.
        btau = max(self.bias_tau, 1.0 / max(2.0, float(self.seq)))
        self._turn_bias += btau * (turn_drive - self._turn_bias)
        turn_drive -= self._turn_bias

        # Сглаживание рулёжки во времени. Это главное место, где рождалось
        # "муха вертится вокруг своей оси". Популяции рулевых нейронов
        # маленькие — десяток нейронов, полтора десятка спайков за тик, —
        # поэтому z-оценка каждого тика это почти чистый шум измерения
        # амплитудой около единицы. Два таких слагаемых (именованные рулевые
        # плюс асимметрия всей нисходящей популяции) дают ±3 КАЖДЫЙ ТИК с
        # независимым знаком: курс делает случайное блуждание шагами под
        # полсотни градусов и проворачивается кругом за семь тиков.
        #
        # У настоящей мухи поворот не переспрашивается заново каждые
        # полсекунды: между саккадами она идёт прямо. Фильтр низких частот
        # делает ровно это — независимый шум усредняется в ноль, устойчивый
        # перекос доживает до поворота. Мёртвая зона добавляет прямые
        # пробежки: пока перекос мал, курс не трогаем вовсе.
        self._turn_lp += self.turn_tau * (turn_drive - self._turn_lp)
        turn_drive = self._turn_lp
        if abs(turn_drive) < self.turn_deadband:
            turn_drive = 0.0

        # Хемотаксис: голодная муха доворачивает на запах. Считается ПОСЛЕ
        # фильтра — это не шум, а направленный сигнал, и сглаживать его
        # незачем.
        chemo = 0.0
        if aim is not None:
            err = self._bearing_error(aim)
            chemo = max(-1.0, min(1.0, err / 60.0)) * self.chemo_gain
            turn_drive += chemo

        if not have_brain and self.spontaneous and spikes:
            # Спонтанная асимметрия собственного шума нисходящих нейронов.
            # Сырое (r-l)/(r+l) на популяциях в десяток нейронов меняет знак
            # КАЖДЫЙ тик: муха дёргается влево-вправо на месте и никуда не
            # идёт. У настоящей мухи спонтанные повороты — это саккады,
            # события длиной в несколько шагов, а не дрожь. Поэтому
            # асимметрию сглаживаем: чередующийся знак усредняется в ноль,
            # устойчивый перекос доживает до поворота.
            l = spikes.get("turn_left", 0)
            r = spikes.get("turn_right", 0)
            if l + r > 0:
                raw = (r - l) / (r + l)
                self._noise_asym += self.noise_tau * (raw - self._noise_asym)
                if abs(self._noise_asym) > 0.08:
                    turn_drive += 0.9 * self._noise_asym
                    self.source = "шум DN"

        # Резервный контур включается, только если рулевых каналов НЕТ
        # вовсе. Проверять сглаженное значение нельзя: в мёртвой зоне оно
        # ноль просто потому, что муха идёт прямо, и резервный контур
        # перехватывал бы управление у живого мозга на каждой прямой.
        if self.fallback and not have_brain and abs(turn_drive) < 1e-3:
            turn_drive, self.source = self._fallback_turn(sensory, total_spikes)
        if ladder_source:
            self.source = ladder_source

        # Рефлекс на удар: если шаг не проходит несколько тиков подряд,
        # муху надо развернуть, иначе она будет толкать стену вечно.
        # Считаем удары, а не «удары в тот же тик, когда шагали»: муха
        # упирается в стену, потом стоит несколько тиков, потом снова
        # упирается — если сбрасывать счётчик на каждом простое, рефлекс
        # не сработает никогда.
        if contact > 0.5:
            self.stuck_for += 1
        elif move != 0:
            self.stuck_for = 0
        if self.fallback and self.stuck_for >= 2:
            # Крутим в одну и ту же сторону несколько ударов подряд: если
            # чередовать каждый раз, муха в углу будет качаться между двумя
            # стенами и никуда не денется.
            turn_drive = 2.6 * self._flip
            self._flip_left -= 1
            if self._flip_left <= 0:
                self._flip = -self._flip
                self._flip_left = 2
            self.stuck_for = 0
            self.source = "удар о препятствие"

        # ---- опьянение --------------------------------------------------
        # Три режима по дозе, ровно как описано у дрозофилы: малая доза
        # даёт гиперактивность и расторможенность, средняя — потерю
        # координации ("шатается"), большая — седацию. Толерантность
        # учитывается на стороне кодировщика и приходит уже в drunk.
        if drunk > 0.02:
            if drunk < 0.3:
                # расторможенность: больше спонтанной ходьбы
                self.arousal = min(1.0, self.arousal + 0.25 * drunk)
                if move == 0 and self.state == "idle":
                    move, self.state = 1, "навеселе"
            elif drunk < 0.72:
                # шатание: случайный доворот, тем сильнее, чем пьянее
                wobble = (self._rng.random() - 0.5) * 2.4 * drunk
                turn_drive += wobble
                self.state = "шатается"
                if self._rng.random() < 0.25 * drunk:
                    move = 0            # запнулась
            else:
                # седация: стоит и не реагирует
                self.state = "отключилась"
                move, run = 0, 0
                act = "halt" if self._rng.random() < 0.3 else None
            if self.source == "мозг":
                self.source = "спирт"

        # Ограничиваем поворот за тик. Без клампа одна большая z-оценка
        # разворачивает муху на пол-оборота за тик, и со стороны это
        # выглядит как волчок, а не как муха.
        turn_drive = max(-self.turn_clamp, min(self.turn_clamp, turn_drive))
        self.heading = (self.heading + self.turn_gain * turn_drive) % 360.0

        # Снапим непрерывный курс на 4 стороны света — с гистерезисом и
        # минимальной выдержкой. Без них курс, зависший у границы секторов,
        # перекидывает dir туда-сюда каждый тик: на экране это и есть
        # "муха вертится вокруг своей оси". Настоящая муха тоже меняет
        # направление рывками, но между рывками идёт прямо.
        self._dir_hold = max(0, self._dir_hold - 1)
        ang = self.heading % 360.0
        off = abs(((ang - self._dir_angle + 180.0) % 360.0) - 180.0)
        urgent = act in ("escape",) or self._refract > 0
        if off > self.heading_snap + self.dir_hysteresis and (
                self._dir_hold == 0 or urgent):
            quad = int(((ang + self.heading_snap) % 360.0) // 90.0)
            self._dir_angle = quad * 90.0
            self._dir_hold = 0 if urgent else self.dir_min_ticks
        dr = DIR_BY_ANGLE[int(self._dir_angle // 90) % 4]

        a = Action(seq=self.seq, dir=dr, move=move, run=run, act=act,
                   part=part, source=self.source)
        a.dbg = {
            "state": self.state,
            "source": self.source,
            "brain_silent_for": self.brain_silent_for,
            "goal": goal.name if goal else None,
            "drive": round(goal.drive, 2) if goal else 0.0,
            "thirst": round(thirst, 2),
            "drunk": round(drunk, 2),
            "climb_urge": round(self.climb_urge, 2),
            "dirt": {k: round(v, 2) for k, v in self._dirt.items()},
            "heading": round(self.heading, 1),
            "arousal": round(self.arousal, 2),
            "turn": round(turn_drive, 2),
            "common": round(common, 2),
            "threat": (None if threat_bearing is None
                       else round(threat_bearing, 0)),
            "z": {k: round(z[k], 2) for k in COMMANDS},
            "hz": {k: round(v, 2) for k, v in rates.items()},
        }
        return a

    # ------------------------------------------------------------------
    def _fallback_turn(self, sensory, total_spikes: int):
        """Рулёжка, когда нисходящие нейроны молчат.

        Сначала пробуем оптомоторный перекос: у мухи курс стабилизируется по
        разнице зрительного потока между глазами — это самое изученное её
        зрительное поведение. Сенсорные нейроны при этом настоящие, молчит
        только путь от них до моторики.

        Если и глаза ничего не говорят, крутим на собственном шуме сети,
        чтобы муха хотя бы обследовала станцию, а не стояла столбом.
        """
        if sensory:
            l = float(sensory.get("vis_left", 0.0))
            r = float(sensory.get("vis_right", 0.0))
            if l + r > 1e-6:
                # Перекос берём относительно скользящей базы, а не в лоб.
                # Иначе постоянная асимметрия картинки крутит муху волчком:
                # разворачиваться надо на ИЗМЕНЕНИЕ обстановки, а к
                # неизменной муха адаптируется, как и её зрительная система.
                dz = self._opto_base.update((r - l) / (l + r))
                if abs(dz) > 0.2:
                    # ограничитель: один тик не должен разворачивать больше
                    # чем примерно на 30 градусов, иначе получается штопор
                    drive = self.optomotor_gain * max(-1.0, min(1.0, dz))
                    return drive, "оптомотор"

        # Саккады. Муха при ходьбе идёт прямо и меняет курс редкими резкими
        # поворотами, а не подкручивает его каждый тик. Мелкая знакопеременная
        # дрожь никуда не уводит: она гасит сама себя.
        if self._saccade_in > 0:
            self._saccade_in -= 1
            return 0.0, "прямой ход"
        if total_spikes > 0:
            mag = 1.2 + 1.6 * ((total_spikes >> 5) % 4) / 3.0
            sign = 1 if ((total_spikes >> 2) & 1) else -1
            self._saccade_in = 4 + (total_spikes >> 7) % 6
            return mag * sign, "саккада"
        return 0.0, "нет сигнала"

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"heading": self.heading, "state": self.state,
                "source": self.source,
                "common": round(getattr(self, "_common", 0.0), 2),
                "brain_silent_for": self.brain_silent_for,
                "arousal": round(self.arousal, 2),
                "z": {k: round(v, 2) for k, v in self._last_z.items()},
                "baseline": {k: round(b.mean, 2) for k, b in self.base.items()}}
