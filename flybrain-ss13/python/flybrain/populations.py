"""Разрешение популяций нейронов по аннотациям FlyWire + ретинотопия.

Задача: превратить regex-описания из config.py в конкретные индексы нейронов,
а для зрительных популяций — ещё и в 2D-раскладку (какой нейрон "смотрит"
в какую точку поля зрения).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .analysis import orient_clusters, split_population
from .config import PopulationSpec
from .connectome import Connectome

# Приоритет при выборе зрительного входа: чем раньше в списке, тем
# периферийнее нейрон. Фоторецепторы в FlyWire реконструированы не полностью,
# поэтому обычно "сетчаткой" становятся ламинарные монополяры L1-L5.
VISUAL_PRIORITY = [
    r"^R[1-8](-\d)?$|^R1-6$|photorecept",   # фоторецепторы, если есть
    r"^L[1-5]$",                             # ламинарные монополярные клетки
    r"^(Mi1|Tm1|Tm2|Tm3|Tm9|Mi4|Mi9)$",      # медулла, если и ламины нет
    # Зрительные проекционные нейроны лобулы: LC4 и LPLC2 — детекторы
    # надвигающегося объекта, они синаптируют прямо на гигантский нейрон
    # DNp01. Ярус далёк от сетчатки, зато до моторики отсюда один-два шага,
    # и на нём модель оживает даже когда периферия реконструирована плохо.
    r"^(LC\d+|LPLC\d+|LPC\d+|LT\d+)",
    r".",                                     # хоть что-то зрительное
]


@dataclass
class Population:
    name: str
    idx: np.ndarray                       # индексы нейронов
    xy: Optional[np.ndarray] = None       # [k,2] в [-1,1], только для зрения
    note: str = ""

    def __len__(self) -> int:
        return len(self.idx)


def _out_weight_per_neuron(cx: Connectome) -> np.ndarray:
    """Суммарный |вес| исходящих связей на нейрон."""
    owner = np.repeat(np.arange(cx.n, dtype=np.int64), np.diff(cx.indptr))
    return np.bincount(owner, weights=np.abs(cx.weights.astype(np.float64)),
                       minlength=cx.n)


def _rx(pattern: Optional[str]):
    if not pattern:
        return None
    return re.compile(pattern, re.IGNORECASE)


def select(cx: Connectome, spec: PopulationSpec) -> np.ndarray:
    """Индексы нейронов, удовлетворяющих спецификации."""
    mask = np.ones(cx.n, dtype=bool)

    rx = _rx(spec.match_super)
    if rx is not None:
        mask &= np.array([bool(rx.search(s)) for s in cx.super_class])

    rx = _rx(spec.match_class)
    if rx is not None:
        mask &= np.array([bool(rx.search(s)) for s in cx.klass])

    rx = _rx(spec.match_type)
    if rx is not None:
        hit = np.array([bool(rx.search(a)) or bool(rx.search(b))
                        for a, b in zip(cx.cell_type, cx.hb_type)])
        mask &= hit

    if spec.side:
        want = spec.side.lower()[0]        # l / r
        mask &= np.array([str(s).lower().startswith(want) for s in cx.side])

    return np.flatnonzero(mask)


# --------------------------------------------------------------------------
def retinotopy(cx: Connectome, idx: np.ndarray) -> np.ndarray:
    """2D-раскладка зрительной популяции.

    Ламина и медулла ретинотопичны, поэтому положение сомы почти напрямую
    задаёт направление взгляда омматидия. Берём главные компоненты облака
    координат и нормируем в квадрат [-1,1]^2.
    """
    if len(idx) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = cx.xyz[idx].astype(np.float64)
    if not np.any(pts):
        # координат нет — раскладываем в квадратную решётку по порядку
        k = int(np.ceil(np.sqrt(len(idx))))
        gx, gy = np.meshgrid(np.linspace(-1, 1, k), np.linspace(-1, 1, k), indexing="ij")
        return np.stack([gx.ravel(), gy.ravel()], 1)[:len(idx)].astype(np.float32)

    pts = pts - pts.mean(0)
    # PCA без scipy
    u, s, vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ vt[:2].T
    for c in range(2):
        lo, hi = np.percentile(proj[:, c], [1, 99])
        rng = max(hi - lo, 1e-9)
        proj[:, c] = np.clip((proj[:, c] - lo) / rng * 2 - 1, -1, 1)
    return proj.astype(np.float32)


# --------------------------------------------------------------------------
@dataclass
class PopulationSet:
    sensory: Dict[str, Population] = field(default_factory=dict)
    motor: Dict[str, Population] = field(default_factory=dict)
    report: List[str] = field(default_factory=list)

    def sensory_idx(self, name: str) -> np.ndarray:
        p = self.sensory.get(name)
        return p.idx if p else np.empty(0, dtype=np.int64)

    def motor_record(self) -> Dict[str, np.ndarray]:
        return {k: v.idx for k, v in self.motor.items() if len(v)}

    def summary(self) -> str:
        lines = ["сенсорные популяции:"]
        for k, v in self.sensory.items():
            lines.append(f"  {k:<16} {len(v):>6} нейронов  {v.note}")
        lines.append("моторные (нисходящие) популяции:")
        for k, v in self.motor.items():
            lines.append(f"  {k:<16} {len(v):>6} нейронов  {v.note}")
        return "\n".join(lines + self.report)


def resolve(cx: Connectome,
            sensory_specs: Dict[str, PopulationSpec],
            motor_specs: Dict[str, PopulationSpec],
            auto_split: bool = True) -> PopulationSet:
    """Собрать все популяции; зрение разрешается с приоритетом слоёв."""
    ps = PopulationSet()

    # ---- зрение -------------------------------------------------------
    # Слой выбирается не по первому совпадению шаблона, а по тому, есть ли
    # у него вообще исходящие связи. В FAFB фоторецепторы реконструированы
    # частично: типы R1-R8 в аннотациях есть, а синапсов у них почти нет,
    # и накачивать их бессмысленно — сигнал упирается в тупик.
    out_w = _out_weight_per_neuron(cx)
    ref = float(np.median(out_w[out_w > 0])) if np.any(out_w > 0) else 0.0
    vis_names = [n for n, sp in sensory_specs.items() if sp.retinotopic]
    candidates = []
    for pattern in VISUAL_PRIORITY:
        rx = re.compile(pattern, re.IGNORECASE)
        trial: Dict[str, np.ndarray] = {}
        ok = True
        for name in vis_names:
            base = select(cx, sensory_specs[name])
            hit = base[[bool(rx.search(cx.cell_type[i]) or rx.search(cx.hb_type[i]))
                        for i in base]] if len(base) else base
            trial[name] = hit
            if len(hit) < 16:              # слишком мало, чтобы быть сетчаткой
                ok = False
        if not ok:
            continue
        allidx = np.concatenate(list(trial.values()))
        score = float(np.median(out_w[allidx])) if len(allidx) else 0.0
        candidates.append((pattern, trial, score, len(allidx)))

    chosen = None
    for pattern, trial, score, size in candidates:
        if ref > 0 and score >= 0.25 * ref:
            chosen = (pattern, trial, score, size)
            break
    if chosen is None and candidates:
        chosen = max(candidates, key=lambda c: c[2])
        ps.report.append(
            "! ни один зрительный слой не выглядит хорошо связанным; "
            "беру лучший из имеющихся")

    if chosen is None:
        for name in vis_names:
            hit = select(cx, sensory_specs[name])
            ps.sensory[name] = Population(name, hit, retinotopy(cx, hit),
                                          note="слой не определён")
        ps.report.append("! зрительный слой выбрать не удалось — беру всё зрительное")
    else:
        pattern, trial, score, size = chosen
        for name, hit in trial.items():
            ps.sensory[name] = Population(name, hit, retinotopy(cx, hit),
                                          note=f"слой {pattern}")
        ps.report.append(
            f"зрительный вход: слой {pattern} ({size} нейронов, "
            f"медианный исходящий вес {score:.1f} мВ при {ref:.1f} по мозгу)")
        skipped = [c for c in candidates if c[0] != pattern and c[2] < 0.25 * ref]
        for pat, _, sc, sz in skipped:
            ps.report.append(
                f"  пропущен слой {pat} ({sz} нейронов): исходящий вес {sc:.1f} мВ — "
                "накачивать его бесполезно, синапсов почти нет")

    # ---- остальная сенсорика -----------------------------------------
    for name, spec in sensory_specs.items():
        if spec.retinotopic:
            continue
        idx = select(cx, spec)
        note = ""
        if len(idx) == 0:
            # Мягкий откат: ищем по классу без уточнения типа. Но только
            # если ограничение по классу или super_class вообще задано —
            # иначе пустая спецификация выбирает ВЕСЬ мозг, и канал
            # превращается в накачку 139 тысяч нейронов разом.
            if spec.match_class or spec.match_super:
                fallback = PopulationSpec(match_class=spec.match_class,
                                          match_super=spec.match_super,
                                          side=spec.side)
                idx = select(cx, fallback)
                note = "откат на весь класс" if len(idx) else "НЕ НАЙДЕНО"
            else:
                note = "НЕ НАЙДЕНО (шаблон типа не совпал, класс не задан)"
        ps.sensory[name] = Population(name, idx, note=note)

    # ---- моторика -----------------------------------------------------
    dn_all = select(cx, motor_specs["dn_all"]) if "dn_all" in motor_specs else np.empty(0, np.int64)
    for name, spec in motor_specs.items():
        idx = select(cx, spec)
        note = ""
        # Слишком мелкая популяция — это шумный сигнал: на трёх нейронах
        # частота квантована ступенями в полгерца. Если задан широкий
        # шаблон, добираем им, но обязательно пишем об этом.
        widen = getattr(spec, "widen", None)
        min_size = getattr(spec, "min_size", 0) or 0
        if widen and len(idx) < min_size:
            wider = PopulationSpec(match_type=widen, match_class=spec.match_class,
                                   match_super=spec.match_super, side=spec.side)
            extra = select(cx, wider)
            # Расширение, забирающее заметную долю всех нисходящих, — это уже
            # не команда, а индикатор общей активности: такой канал будет
            # срабатывать на любой раздражитель.
            cap = max(8, int(0.25 * max(len(dn_all), 1)))
            # Снизу тоже нужен порог. На выгрузке пользователя generic-шаблон
            # дал для feed ровно ТРИ нейрона: такая популяция насыщается от
            # любого входа, и в калибровке канал отвечал +250 Гц и на сахар,
            # и на воду, и на стену. Это выглядит как работающий мозг, но
            # мозгом не является, и лучше честно оставить канал пустым —
            # тогда работает рефлекторная дуга и в dbg видно, что решает
            # не коннектом. Найти настоящий канал в данных умеет
            # scripts/find_channel.py.
            floor = max(4, min_size // 2)
            if len(extra) > len(idx) and len(extra) <= cap and len(extra) >= floor:
                note = (f"расширено шаблоном {widen}: было {len(idx)}, "
                        f"стало {len(extra)}")
                idx = extra
            elif len(extra) > cap:
                note = (f"расширение отклонено: шаблон {widen} даёт {len(extra)} "
                        f"нейронов, это больше четверти всей моторики")
            elif len(extra) and len(extra) < floor:
                note = (f"расширение отклонено: шаблон {widen} даёт всего "
                        f"{len(extra)} нейронов — такой канал отвечает на всё "
                        f"подряд; поищите настоящий через "
                        f"scripts/find_channel.py --for {name}")
                idx = idx[:0]
        if len(idx) == 0 and name != "dn_all":
            note = "НЕ НАЙДЕНО — команда будет неактивна"
        ps.motor[name] = Population(name, idx, note=note)
    if "dn_all" in ps.motor:
        ps.motor["dn_all"] = Population("dn_all", dn_all, note="все нисходящие, для базовой линии")

    # ---- диагностика ---------------------------------------------------
    # Популяции, схлопнувшиеся в один и тот же набор нейронов, неразличимы:
    # сладкое и горькое будут дёргать одно и то же. Если в аннотациях
    # подтипов нет, пробуем разделить набор по связности — куда нейроны
    # проецируются, там и разница.
    groups: Dict[bytes, List[str]] = {}
    for store in (ps.sensory, ps.motor):
        for name, pop in store.items():
            if name in ("dn_all", "dn_left", "dn_right") or not len(pop):
                continue
            groups.setdefault(pop.idx.tobytes(), []).append(name)

    targets = {k: v.idx for k, v in ps.motor.items() if len(v)}
    for names in groups.values():
        if len(names) < 2:
            continue
        sensory_names = [n for n in names if n in ps.sensory]
        affin = {n: (sensory_specs[n].affinity or []) for n in sensory_names
                 if getattr(sensory_specs.get(n), "affinity", None)}
        if auto_split and len(sensory_names) == len(names) and len(affin) >= 2:
            shared = ps.sensory[names[0]].idx
            labels = split_population(cx, shared, k=len(affin), seed=0)
            parts = orient_clusters(cx, shared, labels, affin, targets)
            sizes = {n: len(v) for n, v in parts.items()}
            if all(v >= 4 for v in sizes.values()):
                for n, sub in parts.items():
                    ps.sensory[n] = Population(n, sub, note="выделено по связности")
                left = [n for n in names if n not in parts]
                for n in left:
                    ps.sensory[n] = Population(n, np.empty(0, np.int64),
                                               note="отключено: подтип не выделяется")
                ps.report.append(
                    "! в аннотациях нет подтипов у популяций " + ", ".join(names)
                    + f" — разделил {len(shared)} нейронов по связности на "
                    + ", ".join(f"{n}={sizes[n]}" for n in parts)
                    + (f"; без данных остались: {', '.join(left)}" if left else ""))
                ps.report.append(
                    "  ВНИМАНИЕ: какой кластер на самом деле какой, данные не "
                    "говорят. Ярлыки расставлены по тому, кто сильнее тянет "
                    "свои моторные команды, — это гипотеза, а не аннотация.")
                continue
        ps.report.append(
            "! одинаковый набор нейронов у популяций: " + ", ".join(names)
            + " — эти каналы модель не различает")

    if not np.any(cx.side != ""):
        ps.report.append(
            "! в аннотациях нет сторон тела (колонка side) — левые и правые "
            "популяции совпадут, и муха не сможет поворачивать. Проверьте, "
            "что classification.csv из полной выгрузки Codex.")

    return ps


# --------------------------------------------------------------------------
def build_view_map(pop: Population, grid_w: int, grid_h: int,
                   projection: str = "perspective",
                   eye_height: float = 0.6,
                   d_near: float = 0.75,
                   azim_from_deg: float = -170.0,
                   azim_to_deg: float = 25.0) -> np.ndarray:
    """Для каждого нейрона популяции — индекс клетки поля зрения.

    Из игры приходит квадрат тайлов вокруг персонажа — это карта МИРА, а не
    сетчатка. Глаз мухи — набор направлений взгляда, поэтому между тайлами
    и омматидиями нужна проекция.

    projection="grid" — старое поведение: сетка тайлов кладётся в сетчатку
    один в один. Форма при этом верная (фасеточный глаз и есть 2D-массив
    направлений), но проекция наивная: тайл в одном шаге и тайл в семи
    занимают одинаковое число омматидиев, то есть приближающийся объект не
    растёт. А именно рост объекта — главный зрительный признак для мухи.

    projection="ground" — честная проекция на плоский пол: горизонтальная
    ось сетчатки задаёт азимут, вертикальная — угол места, и точка падения
    луча на пол определяет тайл:

        d = высота глаза / tan(угол места)

    Углы места распределены равномерно, как в настоящем глазу.

    projection="perspective" (по умолчанию) — то же самое, но углы места
    распределены не равномерно, а так, чтобы РАССТОЯНИЕ шло логарифмически
    от d_near до радиуса поля. Причина чисто вычислительная: при честном
    tan-распределении половина омматидиев смотрит ближе одного тайла, и на
    четырёхстах нейронах дальше трёх тайлов глаз не видит НИЧЕГО — из 225
    клеток засеяно два десятка. Логарифмическая раскладка сохраняет главное
    (ближний тайл забирает в 20-40 раз больше омматидиев, чем дальний, то
    есть надвигание кодируется), но при этом ни одна клетка поля не остаётся
    слепой. Это сознательное отступление от оптики ради дискретизации.

    Оси ретинотопии берутся из главных компонент координат сомы, то есть
    определены с точностью до поворота и отражения: какая сторона сетчатки
    смотрит вниз, данные не говорят. На статистику это не влияет — раскладка
    расстояний одна и та же при любом знаке оси, — но какой ИМЕННО нейрон
    смотрит вперёд, модель не знает.
    """
    if pop.xy is None or len(pop) == 0:
        return np.zeros(len(pop), dtype=np.int64)

    if projection == "grid":
        gx = np.clip(((pop.xy[:, 0] + 1) / 2 * grid_w).astype(np.int64), 0, grid_w - 1)
        gy = np.clip(((pop.xy[:, 1] + 1) / 2 * grid_h).astype(np.int64), 0, grid_h - 1)
        return gy * grid_w + gx

    radius = (grid_h - 1) // 2
    d_far = float(radius) + 0.5
    d_near = float(np.clip(d_near, 0.2, d_far * 0.9))

    # Азимут раскладываем по сектору глаза. У дрозофилы поле зрения одного
    # глаза около 190 градусов, а вдвоём они закрывают почти полный круг с
    # перекрытием спереди и сзади — поэтому сектор широкий, а не "полусфера".
    u = (pop.xy[:, 0] + 1.0) / 2.0
    azim = np.radians(azim_from_deg + u * (azim_to_deg - azim_from_deg))

    t = (pop.xy[:, 1] + 1.0) / 2.0                    # 0 = низ сетчатки (близко)
    if projection == "ground":
        phi_near = np.arctan2(eye_height, d_near)
        phi_far = np.arctan2(eye_height, d_far)
        phi = phi_near + t * (phi_far - phi_near)
        dist = eye_height / np.tan(np.clip(phi, 1e-4, np.pi / 2 - 1e-4))
    else:
        dist = d_near * (d_far / d_near) ** t
    dist = np.clip(dist, 0.0, radius)

    forward = dist * np.cos(azim)
    right = dist * np.sin(azim)

    col = np.clip(np.rint(right) + radius, 0, grid_w - 1).astype(np.int64)
    row = np.clip(radius - np.rint(forward), 0, grid_h - 1).astype(np.int64)
    return row * grid_w + col


def eye_sector(name: str, fov_deg: float = 195.0,
               overlap_deg: float = 25.0) -> Tuple[float, float]:
    """Сектор азимутов одного глаза: (от, до) в градусах, 0 — прямо вперёд.

    Левый глаз смотрит от -(fov-overlap) до +overlap, правый зеркально;
    спереди и сзади поля перекрываются, как у дрозофилы.
    """
    fov = float(max(fov_deg, 20.0))
    ov = float(np.clip(overlap_deg, 0.0, fov / 2))
    if name.endswith("left"):
        return -(fov - ov), ov
    return -ov, fov - ov


HEX_SCALE: float = 1.0     # радиус одного омматидия в тех же единицах, что и hex_lattice


def hex_lattice(n_ring: int = 15) -> np.ndarray:
    """Гексагональная решётка омматидиев для визуализации.

    Возвращает [k,2] координат центров в [-1,1]^2. У дрозофилы ~750-800
    омматидиев на глаз; n_ring=15 даёт 721 — ровно тот размер, который
    используют в моделях зрительной системы мухи.
    """
    pts = []
    for q in range(-n_ring, n_ring + 1):
        r1 = max(-n_ring, -q - n_ring)
        r2 = min(n_ring, -q + n_ring)
        for r in range(r1, r2 + 1):
            x = 1.5 * q
            y = np.sqrt(3) * (r + q / 2.0)
            pts.append((x, y))
    a = np.array(pts, dtype=np.float32)
    norm = float(np.abs(a).max())
    a /= norm
    global HEX_SCALE
    HEX_SCALE = 1.0 / norm          # чтобы рисовалка знала размер шестиугольника
    return a
