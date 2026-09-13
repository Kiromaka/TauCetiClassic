"""Разбор коннектома: словарь аннотаций, связность популяций, пути до моторики.

Нужен, когда калибровка показывает пустую таблицу. Симуляция отвечает на
вопрос «доходит ли сигнал», а этот модуль — на вопрос «почему не доходит»:
есть ли у популяции вообще исходящие связи, сколько шагов до нисходящих
нейронов, и кто на самом деле кормит каждый DN.

Всё считается прямо по CSR-матрице, симуляция не запускается.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .connectome import Connectome


# --------------------------------------------------------------------------
def csc(cx: Connectome) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Транспонированный вид: для каждого нейрона — кто в него входит.

    Возвращает (indptr, indices, weights) по ПОСТсинаптическому нейрону.
    """
    n = cx.n
    order = np.argsort(cx.indices, kind="stable")
    post = cx.indices[order].astype(np.int64)
    pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(cx.indptr))[order]
    w = cx.weights[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(post, minlength=n), out=indptr[1:])
    return indptr, pre, w


# --------------------------------------------------------------------------
@dataclass
class PopStats:
    name: str
    size: int
    out_deg_median: float
    out_deg_mean: float
    out_weight_mean: float      # суммарный |вес| исходящих, на нейрон
    in_deg_median: float
    in_weight_mean: float
    dead_frac: float            # доля нейронов вообще без исходящих связей

    def line(self) -> str:
        flag = ""
        if self.dead_frac > 0.5:
            flag = "  <-- больше половины без исходящих связей!"
        elif self.out_weight_mean < 1.0:
            flag = "  <-- почти нечем возбуждать соседей"
        return (f"{self.name:<18}{self.size:>7}"
                f"{self.out_deg_median:>9.0f}{self.out_weight_mean:>11.1f}"
                f"{self.in_deg_median:>9.0f}{self.in_weight_mean:>11.1f}"
                f"{self.dead_frac*100:>8.0f}%{flag}")


def _per_neuron_weight(indptr: np.ndarray, weights: np.ndarray,
                       n: int) -> np.ndarray:
    """Суммарный |вес| на нейрон по CSR, без циклов."""
    owner = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    return np.bincount(owner, weights=np.abs(weights.astype(np.float64)),
                       minlength=n)


def pop_stats(cx: Connectome, name: str, idx: np.ndarray,
              cscv: Optional[Tuple] = None,
              cache: Optional[dict] = None) -> PopStats:
    if len(idx) == 0:
        return PopStats(name, 0, 0, 0, 0, 0, 0, 1.0)
    if cscv is None:
        cscv = csc(cx)
    cptr, _, cw = cscv
    cache = cache if cache is not None else {}
    if "out_w" not in cache:
        cache["out_w"] = _per_neuron_weight(cx.indptr, cx.weights, cx.n)
        cache["in_w"] = _per_neuron_weight(cptr, cw, cx.n)
    out_deg = np.diff(cx.indptr)[idx]
    in_deg = np.diff(cptr)[idx]
    out_w = cache["out_w"][idx]
    in_w = cache["in_w"][idx]
    return PopStats(
        name=name, size=len(idx),
        out_deg_median=float(np.median(out_deg)),
        out_deg_mean=float(out_deg.mean()),
        out_weight_mean=float(out_w.mean()),
        in_deg_median=float(np.median(in_deg)),
        in_weight_mean=float(in_w.mean()),
        dead_frac=float((out_deg == 0).mean()),
    )


STATS_HEADER = (f"{'популяция':<18}{'нейронов':>7}{'исх.степ':>9}"
                f"{'исх.вес':>11}{'вх.степ':>9}{'вх.вес':>11}{'без исх.':>9}")


# --------------------------------------------------------------------------
def influence(cx: Connectome, source: np.ndarray, hops: int = 6,
              signed: bool = False) -> List[np.ndarray]:
    """Куда растекается возбуждение из source за hops шагов.

    Линейное распространение по |W| с нормировкой на каждом шаге: величина
    не физическая, но отвечает на нужный вопрос — какая доля влияния
    доходит до каждого нейрона и на каком шаге.
    """
    n = cx.n
    x = np.zeros(n, dtype=np.float64)
    if len(source) == 0:
        return [x.copy() for _ in range(hops + 1)]
    x[source] = 1.0 / len(source)
    out = [x.copy()]
    w = cx.weights.astype(np.float64) if signed else np.abs(cx.weights).astype(np.float64)
    for _ in range(hops):
        nz = np.flatnonzero(x)
        y = np.zeros(n, dtype=np.float64)
        if nz.size:
            starts = cx.indptr[nz]
            lens = cx.indptr[nz + 1] - starts
            tot = int(lens.sum())
            if tot:
                cum = np.cumsum(lens)
                base = starts.copy()
                base[1:] -= cum[:-1]
                flat = np.repeat(base, lens) + np.arange(tot)
                src_val = np.repeat(x[nz], lens)
                np.add.at(y, cx.indices[flat], w[flat] * src_val)
        s = y.sum()
        if s > 0:
            y /= s                      # нормируем, чтобы сравнивать шаги
        x = y
        out.append(x.copy())
    return out


def reach_table(cx: Connectome,
                sources: Dict[str, np.ndarray],
                targets: Dict[str, np.ndarray],
                hops: int = 6) -> Dict[str, Dict[str, Tuple[float, int]]]:
    """Для каждой пары (сенсорика, моторика): максимум влияния и на каком шаге."""
    res: Dict[str, Dict[str, Tuple[float, int]]] = {}
    for sname, sidx in sources.items():
        if len(sidx) == 0:
            continue
        waves = influence(cx, sidx, hops=hops)
        row: Dict[str, Tuple[float, int]] = {}
        for tname, tidx in targets.items():
            if len(tidx) == 0:
                row[tname] = (0.0, -1)
                continue
            best, best_h = 0.0, -1
            for h, wave in enumerate(waves):
                v = float(wave[tidx].sum())
                if v > best:
                    best, best_h = v, h
            row[tname] = (best, best_h)
        res[sname] = row
    return res


# --------------------------------------------------------------------------
def top_inputs(cx: Connectome, idx: np.ndarray, k: int = 15,
               cscv: Optional[Tuple] = None) -> List[Tuple[str, float, int]]:
    """Какие типы клеток сильнее всего кормят эту популяцию.

    Возвращает список (тип, суммарный |вес|, число нейронов-источников),
    отсортированный по весу. Это прямой ответ на вопрос «что дёргать,
    чтобы завести эту команду».
    """
    if len(idx) == 0:
        return []
    if cscv is None:
        cscv = csc(cx)
    cptr, cpre, cw = cscv
    acc: Dict[str, float] = collections.defaultdict(float)
    cnt: Dict[str, set] = collections.defaultdict(set)
    for i in idx:
        a, b = cptr[i], cptr[i + 1]
        for src, weight in zip(cpre[a:b], cw[a:b]):
            t = cx.cell_type[src] or cx.klass[src] or cx.super_class[src] or "?"
            acc[t] += abs(float(weight))
            cnt[t].add(int(src))
    rows = [(t, v, len(cnt[t])) for t, v in acc.items()]
    rows.sort(key=lambda r: -r[1])
    return rows[:k]


def top_outputs(cx: Connectome, idx: np.ndarray, k: int = 15
                ) -> List[Tuple[str, float, int]]:
    """Кому эта популяция что-то отдаёт."""
    if len(idx) == 0:
        return []
    acc: Dict[str, float] = collections.defaultdict(float)
    cnt: Dict[str, set] = collections.defaultdict(set)
    for i in idx:
        a, b = cx.indptr[i], cx.indptr[i + 1]
        for dst, weight in zip(cx.indices[a:b], cx.weights[a:b]):
            t = cx.cell_type[dst] or cx.klass[dst] or cx.super_class[dst] or "?"
            acc[t] += abs(float(weight))
            cnt[t].add(int(dst))
    rows = [(t, v, len(cnt[t])) for t, v in acc.items()]
    rows.sort(key=lambda r: -r[1])
    return rows[:k]


# --------------------------------------------------------------------------
def vocabulary(cx: Connectome, top: int = 25) -> str:
    """Что вообще написано в аннотациях этого снапшота.

    Дефолтные шаблоны популяций писались под один словарь типов, а в
    выгрузке может оказаться другой. Это распечатка настоящего словаря.
    """
    lines: List[str] = []

    def counts(arr, title, limit=top):
        vals, cnts = np.unique(arr, return_counts=True)
        pairs = sorted(zip(vals, cnts), key=lambda t: -t[1])
        lines.append(f"\n{title} ({len(pairs)} различных):")
        for v, c in pairs[:limit]:
            lines.append(f"    {c:>7}  {v or '(пусто)'}")
        if len(pairs) > limit:
            lines.append(f"    … и ещё {len(pairs)-limit}")

    counts(cx.super_class, "super_class")
    counts(cx.klass, "class")
    counts(cx.nt, "нейромедиатор", limit=12)
    counts(cx.side, "side", limit=8)

    lines.append(f"\nтипов клеток различных: "
                 f"{len(np.unique(cx.cell_type[cx.cell_type != '']))}, "
                 f"без типа: {int((cx.cell_type == '').sum())}")

    # типы внутри интересных нам классов
    for label, mask in (
        ("нисходящие (super_class descending)", cx.super_class == "descending"),
        ("моторные", cx.super_class == "motor"),
        ("сенсорные", np.isin(cx.super_class, ["sensory", "sensory_ascending"])),
        ("зрительные проекционные", cx.super_class == "visual_projection"),
    ):
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        vals, cnts = np.unique(cx.cell_type[idx], return_counts=True)
        pairs = sorted(zip(vals, cnts), key=lambda t: -t[1])
        lines.append(f"\nтипы клеток внутри «{label}» — всего {len(idx)} нейронов, "
                     f"{len(pairs)} типов:")
        shown = [f"{v}({c})" for v, c in pairs[:60] if v]
        lines.append("    " + ", ".join(shown))
        if len(pairs) > 60:
            lines.append(f"    … и ещё {len(pairs)-60}")
    return "\n".join(lines)


def grep_types(cx: Connectome, pattern: str, limit: int = 60) -> List[Tuple[str, int]]:
    """Найти типы клеток по шаблону — проверить гипотезу об именовании."""
    rx = re.compile(pattern, re.IGNORECASE)
    hit = collections.Counter()
    for t, h in zip(cx.cell_type, cx.hb_type):
        if (t and rx.search(t)) or (h and rx.search(h)):
            hit[t or h] += 1
    return hit.most_common(limit)


# --------------------------------------------------------------------------
def split_population(cx: Connectome, idx: np.ndarray, k: int = 2,
                     seed: int = 0, iters: int = 40) -> np.ndarray:
    """Разбить популяцию на k групп по тому, КУДА она проецируется.

    Нужно, когда в аннотациях нет подтипов: например, все вкусовые нейроны
    свалены в один класс `gustatory` без деления на сладкое и горькое. Но
    сладкие и горькие GRN у мухи идут в разные нижестоящие цепи, и это видно
    прямо в матрице связей.

    Кластеризация: случайная проекция строк смежности в 64 измерения
    (лемма Джонсона-Линденштрауса) плюс k-means на косинусной метрике.
    Без scipy и sklearn, чтобы не тащить зависимости в рантайм.

    Возвращает массив меток 0..k-1 длиной len(idx).
    """
    rng = np.random.default_rng(seed)
    m = len(idx)
    if m == 0:
        return np.empty(0, dtype=np.int64)
    if m <= k:
        return np.arange(m) % k

    dim = 64
    proj = rng.normal(0, 1.0 / np.sqrt(dim), size=(cx.n, dim)).astype(np.float32)
    feat = np.zeros((m, dim), dtype=np.float32)
    for r, i in enumerate(idx):
        a, b = cx.indptr[i], cx.indptr[i + 1]
        if b > a:
            w = np.abs(cx.weights[a:b]).astype(np.float32)
            feat[r] = w @ proj[cx.indices[a:b]]
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    feat /= norm                                   # косинусная метрика

    # k-means++ на инициализацию, иначе на вырожденных данных всё схлопывается
    centers = np.empty((k, dim), dtype=np.float32)
    centers[0] = feat[rng.integers(m)]
    for c in range(1, k):
        d = 1.0 - feat @ centers[:c].T
        d = d.min(axis=1) ** 2
        total = d.sum()
        centers[c] = feat[rng.integers(m) if total <= 0
                          else int(np.searchsorted(np.cumsum(d / total),
                                                   rng.random()))]
    labels = np.zeros(m, dtype=np.int64)
    for _ in range(iters):
        new = np.argmax(feat @ centers.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            sel = labels == c
            if sel.any():
                v = feat[sel].mean(axis=0)
                nv = np.linalg.norm(v)
                if nv > 0:
                    centers[c] = v / nv
    return labels


def orient_clusters(cx: Connectome, idx: np.ndarray, labels: np.ndarray,
                    affinities: Dict[str, Sequence[str]],
                    targets: Dict[str, np.ndarray],
                    hops: int = 4) -> Dict[str, np.ndarray]:
    """Раздать кластеры по именам популяций, глядя на их влияние.

    affinities — какие моторные команды «должна» тянуть каждая популяция
    (сладкое к питанию, горькое к заднему ходу). Считаем влияние каждого
    кластера на эти команды и раздаём кластеры так, чтобы суммарное
    соответствие было максимальным.

    Ярлыки при этом ВЫВЕДЕНЫ из связности, а не взяты из аннотаций: какой
    кластер действительно сладкий, данные не говорят. Об этом обязательно
    надо писать в отчёте.
    """
    names = list(affinities)
    k = int(labels.max()) + 1 if len(labels) else 0
    if k == 0 or not names:
        return {}

    score = np.zeros((k, len(names)))
    for c in range(k):
        members = idx[labels == c]
        if not len(members):
            continue
        waves = influence(cx, members, hops=hops)
        for j, nm in enumerate(names):
            tot = 0.0
            for tname in affinities[nm]:
                t = targets.get(tname)
                if t is None or not len(t):
                    continue
                tot += max(float(w[t].sum()) for w in waves)
            score[c, j] = tot

    # кластеров и имён немного, поэтому перебираем назначения честно
    import itertools
    best, best_val = None, -1.0
    for perm in itertools.permutations(range(k), min(k, len(names))):
        val = sum(score[perm[j], j] for j in range(len(perm)))
        if val > best_val:
            best_val, best = val, perm
    out: Dict[str, np.ndarray] = {}
    for j, c in enumerate(best):
        out[names[j]] = idx[labels == c]
    for nm in names[len(best):]:
        out[nm] = np.empty(0, dtype=np.int64)
    return out


# --------------------------------------------------------------------------
def shuffle_connectome(cx: Connectome, seed: int = 0,
                       keep_weights: bool = True) -> Connectome:
    """Нулевая модель: та же статистика, разрушенная структура.

    Сохраняем исходящую степень каждого нейрона и распределение весов, но
    перемешиваем, КУДА идут связи. Получается мозг с той же «физикой» и
    той же плотностью, но без единого настоящего пути.

    Нужно, чтобы отвечать на вопрос «а точно ли работает коннектом»: если
    поведение мухи на настоящем коннектоме не отличается от поведения на
    перемешанном, значит работает не коннектом, а всё остальное.
    """
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, cx.n, size=cx.nnz).astype(np.int32)
    weights = cx.weights.copy()
    if not keep_weights:
        rng.shuffle(weights)
    return Connectome(
        indptr=cx.indptr.copy(), indices=indices, weights=weights,
        root_ids=cx.root_ids, cell_type=cx.cell_type, hb_type=cx.hb_type,
        klass=cx.klass, super_class=cx.super_class, side=cx.side,
        nt=cx.nt, xyz=cx.xyz,
        meta={**cx.meta, "shuffled": True, "shuffle_seed": seed},
    )
