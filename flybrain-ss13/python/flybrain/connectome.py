"""Загрузка и препроцессинг коннектома FlyWire (FAFB, снапшот 783).

Вход — выгрузка Codex:
    connections.csv.gz              pre_root_id, post_root_id, neuropil,
                                    syn_count, nt_type
    classification.csv.gz           root_id, flow, super_class, class,
                                    sub_class, hemilineage, side, nerve
    consolidated_cell_types.csv.gz  root_id, primary_type, additional_type(s)
    coordinates.csv.gz              root_id, position (для ретинотопии)

Выход — один .npz: CSR-матрица по пресинаптическому нейрону + аннотации.

ВАЖНО про типы клеток. В снапшотах до 630 они лежали в classification.csv
в колонке cell_type, в свежих её там нет — типы уехали в
consolidated_cell_types.csv. А без типов не найти ни DNp09, ни MDN, ни Gr5a,
то есть ни моторику, ни половину сенсорики. Поэтому загрузчик перебирает
все переданные ему файлы, пока не наберёт типы, и падает с внятной
ошибкой, если не набрал.

Состав и имена колонок вообще плавают от снапшота к снапшоту, так что всё
разрешается через таблицу псевдонимов, а недостающие необязательные
колонки просто заполняются пустыми значениями.
"""
from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .config import ModelParams

# --------------------------------------------------------------------------
COL_ALIASES = {
    "pre":       ("pre_root_id", "pre_pt_root_id", "pre_id", "presynaptic", "source"),
    "post":      ("post_root_id", "post_pt_root_id", "post_id", "postsynaptic", "target"),
    "syn":       ("syn_count", "synapses", "count", "weight", "n_syn"),
    "nt":        ("nt_type", "neurotransmitter", "nt", "top_nt"),
    "root":      ("root_id", "id", "pt_root_id"),
    "cell_type": ("cell_type", "primary_type", "type", "consensus_type",
                  "cell type", "celltype", "malecns_type", "consolidated_cell_type"),
    "hb_type":   ("hemibrain_type", "hemibrain", "hb_type",
                  "additional_type(s)", "additional_types", "additional_type",
                  "synonyms", "supertype"),
    "klass":     ("class", "cell_class", "cell class"),
    "super":     ("super_class", "superclass", "super class"),
    "side":      ("side", "hemisphere"),
    "flow":      ("flow",),
    "x":         ("x", "pos_x", "soma_x"),
    "y":         ("y", "pos_y", "soma_y"),
    "z":         ("z", "pos_z", "soma_z"),
    "position":  ("position", "pt_position", "soma_position", "coordinates"),
}


def _pick(header: Sequence[str], key: str) -> Optional[int]:
    low = [h.strip().strip('"').lower() for h in header]
    for alias in COL_ALIASES[key]:
        if alias in low:
            return low.index(alias)
    return None


def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def columns_of(path: str) -> List[str]:
    """Заголовок CSV — пригодится, чтобы понять, что вообще пришло в выгрузке."""
    import csv as _csv
    with _open(path) as fh:
        return [h.strip().strip('"') for h in next(_csv.reader(fh))]


def _read_csv(path: str, want: Iterable[str],
              optional: Iterable[str] = ()) -> Dict[str, Optional[list]]:
    """Мини-ридер CSV, чтобы не тащить pandas в рантайм.

    Колонки из optional, которых в файле нет, возвращаются как None:
    состав выгрузки FlyWire меняется от снапшота к снапшоту, и жёсткое
    требование колонки ломает загрузку на ровном месте.
    """
    import csv as _csv

    want = list(want)
    optional = set(optional)
    out: Dict[str, Optional[list]] = {k: [] for k in want}
    with _open(path) as fh:
        reader = _csv.reader(fh)
        header = next(reader)
        idx = {k: _pick(header, k) for k in want}
        missing = [k for k, v in idx.items() if v is None and k not in optional]
        if missing:
            raise KeyError(
                f"в {os.path.basename(path)} не найдены обязательные колонки "
                f"{missing}; есть: {header}")
        present = [k for k in want if idx[k] is not None]
        for k in want:
            if idx[k] is None:
                out[k] = None
        width = max((idx[k] for k in present), default=-1) + 1
        for row in reader:
            if len(row) < width:
                continue
            for k in present:
                out[k].append(row[idx[k]])
    return out


# --------------------------------------------------------------------------
@dataclass
class Connectome:
    """CSR по пресинаптическому нейрону.

    indices[indptr[i]:indptr[i+1]] — постсинаптические индексы нейрона i
    weights[...] — прибавка к синаптической переменной g в мВ (со знаком)
    """
    indptr: np.ndarray      # int64  [n+1]
    indices: np.ndarray     # int32  [nnz]
    weights: np.ndarray     # float32[nnz]
    root_ids: np.ndarray    # uint64 [n]
    cell_type: np.ndarray   # <U     [n]
    hb_type: np.ndarray     # <U     [n]
    klass: np.ndarray       # <U     [n]
    super_class: np.ndarray # <U     [n]
    side: np.ndarray        # <U     [n]
    nt: np.ndarray          # <U     [n]
    xyz: np.ndarray         # float32[n,3]  (нули, если координат не было)
    meta: dict

    @property
    def n(self) -> int:
        return len(self.root_ids)

    @property
    def nnz(self) -> int:
        return len(self.indices)

    def out_degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            indptr=self.indptr, indices=self.indices, weights=self.weights,
            root_ids=self.root_ids, cell_type=self.cell_type, hb_type=self.hb_type,
            klass=self.klass, super_class=self.super_class, side=self.side,
            nt=self.nt, xyz=self.xyz,
            meta=np.array(repr(self.meta), dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "Connectome":
        z = np.load(path, allow_pickle=True)
        meta = {}
        try:
            meta = eval(str(z["meta"]))  # noqa: S307 — свой же файл
        except Exception:
            pass
        return cls(
            indptr=z["indptr"], indices=z["indices"], weights=z["weights"],
            root_ids=z["root_ids"], cell_type=z["cell_type"], hb_type=z["hb_type"],
            klass=z["klass"], super_class=z["super_class"], side=z["side"],
            nt=z["nt"], xyz=z["xyz"], meta=meta,
        )

    # ------------------------------------------------------------------
    def describe(self) -> str:
        deg = self.out_degree()
        sup, cnt = np.unique(self.super_class, return_counts=True)
        top = ", ".join(f"{s or '?'}={c}" for s, c in
                        sorted(zip(sup, cnt), key=lambda t: -t[1])[:8])
        return (f"нейронов {self.n}, связей {self.nnz}, "
                f"средняя исходящая степень {deg.mean():.1f} (макс {deg.max()}), "
                f"вес |w| median {np.median(np.abs(self.weights)):.3f} мВ\n"
                f"super_class: {top}")


# --------------------------------------------------------------------------
def build(connections_csv: str,
          classification_csv: str,
          coordinates_csv: Optional[str] = None,
          consolidated_csv: Optional[str] = None,
          type_csvs: Optional[Sequence[str]] = None,
          params: Optional[ModelParams] = None,
          verbose: bool = True) -> Connectome:
    """Собрать Connectome из выгрузки Codex.

    type_csvs — дополнительные файлы, откуда брать типы клеток. В свежих
    снапшотах в classification.csv колонки cell_type уже нет, типы уехали
    в consolidated_cell_types.csv, поэтому источники перебираются по очереди.
    """
    params = params or ModelParams()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    # ---- аннотации ---------------------------------------------------
    log("читаю classification…")
    log(f"  колонки: {columns_of(classification_csv)}")
    ann_cols = ["root", "cell_type", "hb_type", "klass", "super", "side"]
    ann = _read_csv(classification_csv, ann_cols,
                    optional=["cell_type", "hb_type", "klass", "super", "side"])
    n_rows = len(ann["root"])
    for key in ann_cols[1:]:
        if ann[key] is None:
            ann[key] = [""] * n_rows

    root_ids = np.array([int(r) for r in ann["root"]], dtype=np.uint64)
    order = np.argsort(root_ids)
    root_ids = root_ids[order]
    index_of = {int(r): i for i, r in enumerate(root_ids)}
    n = len(root_ids)
    log(f"  нейронов в аннотациях: {n}")

    def col(key: str) -> np.ndarray:
        arr = np.array(ann[key], dtype=object)[order]
        return np.array([str(v).strip() for v in arr], dtype="<U48")

    cell_type = col("cell_type")
    hb_type = col("hb_type")
    klass = col("klass")
    super_class = col("super")
    side = col("side")

    # ---- типы клеток --------------------------------------------------
    # Без них не найти ни DNp09, ни MDN, ни Gr5a, то есть ни моторику,
    # ни половину сенсорики. В классификации их может не быть вовсе.
    sources: List[str] = []
    if consolidated_csv:
        sources.append(consolidated_csv)
    sources.extend(type_csvs or [])
    seen = set()
    for path in sources:
        if not path or not os.path.exists(path):
            continue
        real = os.path.realpath(path)
        if real in seen or real == os.path.realpath(classification_csv):
            continue
        seen.add(real)
        try:
            cols = columns_of(path)
        except Exception as exc:
            log(f"  пропускаю {os.path.basename(path)}: {exc}")
            continue
        if _pick(cols, "cell_type") is None and _pick(cols, "hb_type") is None:
            continue
        log(f"читаю типы из {os.path.basename(path)} (колонки: {cols})…")
        try:
            t = _read_csv(path, ["root", "cell_type", "hb_type"],
                          optional=["cell_type", "hb_type"])
        except KeyError as exc:
            log(f"  пропускаю: {exc}")
            continue
        added_primary = added_alt = 0
        prim = t["cell_type"] or []
        alt = t["hb_type"] or []
        for k, r in enumerate(t["root"]):
            try:
                i = index_of.get(int(r))
            except ValueError:
                continue
            if i is None:
                continue
            if k < len(prim) and prim[k] and not cell_type[i]:
                cell_type[i] = str(prim[k]).strip()[:48]
                added_primary += 1
            if k < len(alt) and alt[k] and not hb_type[i]:
                hb_type[i] = str(alt[k]).strip()[:48]
                added_alt += 1
        log(f"  проставлено типов: {added_primary}, доп. типов: {added_alt}")

    typed = int(np.count_nonzero(cell_type != ""))
    log(f"нейронов с известным типом: {typed} из {n}")
    if typed == 0:
        raise RuntimeError(
            "ни в одном файле не нашлись типы клеток (cell_type / primary_type).\n"
            "В свежих снапшотах FlyWire их нет в classification.csv — они лежат\n"
            "в consolidated_cell_types.csv.gz. Скачайте его:\n"
            "    python3 scripts/fetch_connectome.py --all\n"
            "и повторите сборку. Без типов не найти ни нисходящие нейроны\n"
            "(DNp09, MDN, DNa01), ни вкусовые и обонятельные популяции.")

    # ---- координаты для ретинотопии ----------------------------------
    xyz = np.zeros((n, 3), dtype=np.float32)
    if coordinates_csv and os.path.exists(coordinates_csv):
        log("читаю coordinates…")
        try:
            crd = _read_csv(coordinates_csv, ["root", "position"])
            pat = re.compile(r"-?\d+\.?\d*")
            for r, p in zip(crd["root"], crd["position"]):
                i = index_of.get(int(r))
                if i is None or xyz[i].any():
                    continue
                nums = pat.findall(p)
                if len(nums) >= 3:
                    xyz[i] = [float(nums[0]), float(nums[1]), float(nums[2])]
        except KeyError:
            crd = _read_csv(coordinates_csv, ["root", "x", "y", "z"])
            for r, a, b, c in zip(crd["root"], crd["x"], crd["y"], crd["z"]):
                i = index_of.get(int(r))
                if i is not None and not xyz[i].any():
                    xyz[i] = [float(a), float(b), float(c)]

    # ---- связи --------------------------------------------------------
    log("читаю connections… (это самый долгий шаг, файл ~100-300 МБ)")
    log(f"  колонки: {columns_of(connections_csv)}")
    con = _read_csv(connections_csv, ["pre", "post", "syn", "nt"], optional=["nt"])
    pre_raw = np.array(con["pre"], dtype=np.uint64)
    post_raw = np.array(con["post"], dtype=np.uint64)
    syn = np.array(con["syn"], dtype=np.float64)
    if con["nt"] is None:
        log("  колонки nt_type нет — беру медиатор из аннотаций нейронов")
        nt_by_root = _nt_from_annotations(classification_csv, type_csvs or [], index_of, n)
        pre_tmp = np.searchsorted(root_ids, pre_raw)
        pre_tmp = np.clip(pre_tmp, 0, n - 1)
        nt_raw = nt_by_root[pre_tmp]
    else:
        nt_raw = np.array([s.strip().upper() for s in con["nt"]], dtype="<U16")
    del con
    log(f"  строк связей: {len(pre_raw)}")

    # root_id -> плотный индекс через searchsorted (root_ids отсортированы)
    def to_index(arr: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(root_ids, arr)
        pos = np.clip(pos, 0, n - 1)
        bad = root_ids[pos] != arr
        pos = pos.astype(np.int64)
        pos[bad] = -1
        return pos

    pre = to_index(pre_raw)
    post = to_index(post_raw)
    keep = (pre >= 0) & (post >= 0) & (syn >= params.min_syn_count)
    dropped = int((~keep).sum())
    pre, post, syn, nt_edge = pre[keep], post[keep], syn[keep], nt_raw[keep]
    log(f"  отброшено {dropped} строк (нет в аннотациях или syn < {params.min_syn_count})")

    # нейромедиатор нейрона = мода по его исходящим связям
    nt_per_neuron = _majority_nt(pre, nt_edge, n)

    # схлопываем нейропили: одна пара (pre,post) — одна связь
    log("агрегирую по нейропилям…")
    key = pre.astype(np.int64) * np.int64(n) + post.astype(np.int64)
    ordk = np.argsort(key, kind="stable")
    key, pre, post, syn = key[ordk], pre[ordk], post[ordk], syn[ordk]
    uniq_start = np.empty(len(key), dtype=bool)
    uniq_start[0] = True
    np.not_equal(key[1:], key[:-1], out=uniq_start[1:])
    grp = np.cumsum(uniq_start) - 1
    syn_sum = np.bincount(grp, weights=syn)
    pre_u = pre[uniq_start]
    post_u = post[uniq_start]

    # знак и вес
    signs = np.array([params.nt_sign.get(str(t), 0.0) for t in nt_per_neuron],
                     dtype=np.float32)
    w = (syn_sum.astype(np.float32)
         * signs[pre_u]
         * np.float32(params.w_synapse)
         * np.float32(params.gain))
    nz = w != 0
    pre_u, post_u, w = pre_u[nz], post_u[nz], w[nz]
    log(f"  уникальных связей: {len(w)} (нулевых по НТ отброшено {int((~nz).sum())})")

    # CSR по pre
    ordp = np.argsort(pre_u, kind="stable")
    pre_u, post_u, w = pre_u[ordp], post_u[ordp], w[ordp]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre_u, minlength=n), out=indptr[1:])

    cx = Connectome(
        indptr=indptr,
        indices=post_u.astype(np.int32),
        weights=w.astype(np.float32),
        root_ids=root_ids,
        cell_type=cell_type, hb_type=hb_type, klass=klass,
        super_class=super_class, side=side,
        nt=nt_per_neuron.astype("<U16"),
        xyz=xyz,
        meta={"source": os.path.basename(connections_csv),
              "gain": params.gain, "w_synapse": params.w_synapse,
              "min_syn_count": params.min_syn_count,
              "nt_sign": dict(params.nt_sign)},
    )
    log(cx.describe())
    return cx


TYPE_FILE_HINTS = ("consolidated_cell_types", "cell_types", "visual_neuron_types",
                   "neurons", "classification", "annotations", "types")


def find_type_sources(data_dir: str) -> List[str]:
    """Все CSV в каталоге, где есть root_id и хоть какая-то колонка типа."""
    out = []
    if not os.path.isdir(data_dir):
        return out
    for name in sorted(os.listdir(data_dir)):
        if not (name.endswith(".csv") or name.endswith(".csv.gz")):
            continue
        path = os.path.join(data_dir, name)
        try:
            header = columns_of(path)
        except Exception:
            continue
        if _pick(header, "root") is None:
            continue
        if _pick(header, "cell_type") is not None or _pick(header, "hb_type") is not None:
            out.append(path)
    # сначала файлы с говорящими именами
    out.sort(key=lambda p: next((i for i, h in enumerate(TYPE_FILE_HINTS)
                                 if h in os.path.basename(p)), len(TYPE_FILE_HINTS)))
    return out


def _has(path: str, key: str) -> bool:
    import csv as _csv
    with _open(path) as fh:
        header = next(_csv.reader(fh))
    return _pick(header, key) is not None


def _nt_from_annotations(classification_csv: str, extra: Sequence[str],
                         index_of: dict, n: int) -> np.ndarray:
    """Медиатор по нейронам, если его нет в таблице связей."""
    out = np.full(n, "", dtype="<U16")
    for path in [classification_csv, *extra]:
        if not path or not os.path.exists(path):
            continue
        try:
            cols = columns_of(path)
        except Exception:
            continue
        if _pick(cols, "nt") is None:
            continue
        t = _read_csv(path, ["root", "nt"], optional=["nt"])
        if t["nt"] is None:
            continue
        for r, v in zip(t["root"], t["nt"]):
            try:
                i = index_of.get(int(r))
            except ValueError:
                continue
            if i is not None and v and not out[i]:
                out[i] = str(v).strip().upper()[:16]
        if np.count_nonzero(out != ""):
            break
    return out


def _majority_nt(pre: np.ndarray, nt_edge: np.ndarray, n: int) -> np.ndarray:
    """Нейромедиатор нейрона = самый частый среди его исходящих связей."""
    labels, codes = np.unique(nt_edge, return_inverse=True)
    tally = np.zeros((n, len(labels)), dtype=np.int32)
    np.add.at(tally, (pre, codes), 1)
    best = tally.argmax(axis=1)
    empty = tally.sum(axis=1) == 0
    out = labels[best]
    out[empty] = ""
    return out


# --------------------------------------------------------------------------
def make_synthetic(n: int = 20000, avg_degree: int = 100, seed: int = 0,
                   params: Optional[ModelParams] = None,
                   structured: bool = True) -> Connectome:
    """Синтетический коннектом со статистикой, похожей на FlyWire.

    Нужен для тестов и для бенчмарка без скачивания сотен мегабайт данных.
    Раскладка популяций (типы, стороны, координаты) имитирует настоящую,
    поэтому весь пайплайн — популяции, кодировщики, декодер — работает.

    structured=True добавляет слоистый путь
        сенсорика -> слой 1 -> слой 2 -> нисходящие нейроны
    с латерализацией (левое зрение тянет за левый поворот). Это НЕ биология,
    а тестовая заглушка: в настоящем коннектоме такие пути есть на самом деле,
    и без них случайный граф просто рассеивает сигнал и муха стоит столбом.
    structured=False даёт чисто случайный граф — им удобно мерить
    производительность движка.
    """
    params = params or ModelParams()
    rng = np.random.default_rng(seed)

    # ---- разметка популяций -------------------------------------------
    cell_type = np.empty(n, dtype="<U48")
    klass = np.empty(n, dtype="<U48")
    super_class = np.empty(n, dtype="<U48")
    side = np.empty(n, dtype="<U48")
    cell_type[:] = ""
    klass[:] = ""
    super_class[:] = "central"
    side[:] = ""
    xyz = rng.normal(0, 100, (n, 3)).astype(np.float32)

    groups: Dict[str, np.ndarray] = {}
    cur = 0

    def assign(count, ctype, cls, sup, sd, key=None, retino=False):
        nonlocal cur
        count = min(count, n - cur)
        sl = np.arange(cur, cur + count)
        cell_type[sl] = ctype
        klass[sl] = cls
        super_class[sl] = sup
        side[sl] = sd
        if retino:
            k = int(np.ceil(np.sqrt(count)))
            gx, gy = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
            pts = np.stack([gx.ravel(), gy.ravel()], 1)[:count].astype(np.float32)
            # Оптические доли сидят по бокам от центрального мозга. Раньше
            # отступ был 5000 при разбросе центрального мозга 100 — то есть
            # "глаза" улетали на полсотни ширин мозга, и любая раскладка по
            # реальным координатам (3D-панель, ретинотопия) вырождалась в
            # две точки где-то на горизонте. Держим пропорции мушиными:
            # доля шире центра примерно вдвое.
            span = pts.max() * 10 if count else 1.0
            xyz[sl, 0] = (pts[:, 0] * 10 - span / 2
                          + (-1 if sd == "left" else 1) * 230.0)
            xyz[sl, 1] = pts[:, 1] * 10 - span / 2
            xyz[sl, 2] = rng.normal(0, 12, count)
        if key:
            groups[key] = np.concatenate([groups[key], sl]) if key in groups else sl
        cur += count

    assign(400, "L1", "visual", "optic", "left", "vis_l", retino=True)
    assign(400, "L1", "visual", "optic", "right", "vis_r", retino=True)
    assign(60, "ORN_DM1", "olfactory", "sensory", "left", "olf_good")
    assign(60, "ORN_DA2", "olfactory", "sensory", "left", "olf_bad")
    assign(30, "ORN_V", "olfactory", "sensory", "left", "olf_co2")
    assign(40, "Gr5a", "gustatory", "sensory", "left", "sugar")
    assign(40, "Gr66a", "gustatory", "sensory", "left", "bitter")
    assign(20, "ppk28", "gustatory", "sensory", "left", "water")
    assign(120, "JO-B", "mechanosensory", "sensory", "left", "jo")
    assign(120, "bristle", "mechanosensory", "sensory", "right", "bristle")
    assign(20, "HC", "thermosensory", "sensory", "left", "hot")
    assign(20, "CC", "thermosensory", "sensory", "right", "cold")
    # Восходящие: ноцицепция и проприоцепция. Без них канал боли на
    # синтетике вообще не находил нейронов, побег не мог сработать в
    # принципе, и поведенческая проба по удару мерила случайную прогулку.
    assign(60, "ascending_nocic", "ascending", "ascending", "left", "nocic")

    for sd in ("left", "right"):
        assign(2, "DNp01", "descending", "descending", sd, "escape")
        assign(2, "DNp09", "descending", "descending", sd, "forward")
        assign(2, "MDN", "descending", "descending", sd, "backward")
        assign(4, "DNa01", "descending", "descending", sd, "turn_" + sd)
        assign(4, "DNa02", "descending", "descending", sd, "turn_" + sd)
    assign(6, "DNg11", "descending", "descending", "left", "groom")
    assign(6, "DNp02", "descending", "descending", "right", "threat")
    assign(8, "MN9", "motor", "motor", "left", "feed")
    assign(4, "DNa08", "descending", "descending", "left", "speed")

    sensory_end = cur
    # два промежуточных слоя (аналоги медуллы/лобулы) и остальной мозг
    l1_n = min(2400, max(200, (n - cur) // 8))
    l2_n = min(1200, max(100, (n - cur) // 16))
    l1 = np.arange(cur, cur + l1_n); cur += l1_n
    l2 = np.arange(cur, cur + l2_n); cur += l2_n
    rest = np.arange(cur, n)

    # ---- нейромедиаторы ------------------------------------------------
    nt_choices = np.array(["ACH", "GABA", "GLUT", "DA", "SER", "OCT"])
    nt = rng.choice(nt_choices, size=n, p=[0.55, 0.20, 0.18, 0.03, 0.02, 0.02])
    nt[:sensory_end] = "ACH"          # сенсорика возбуждающая
    nt[l1] = np.where(rng.random(len(l1)) < 0.65, "ACH", "GABA")
    nt[l2] = np.where(rng.random(len(l2)) < 0.65, "ACH", "GABA")
    signs = np.array([params.nt_sign.get(t, 0.0) for t in nt], dtype=np.float32)

    # ---- рёбра ---------------------------------------------------------
    src_list: List[np.ndarray] = []
    dst_list: List[np.ndarray] = []
    syn_list: List[np.ndarray] = []

    def wire(src, dst, per_src, syn_lo=20, syn_hi=90):
        """Плотный сходящийся пучок src -> dst."""
        if len(src) == 0 or len(dst) == 0 or per_src <= 0:
            return
        s = np.repeat(src, per_src)
        d = rng.choice(dst, size=len(s))
        y = rng.uniform(syn_lo, syn_hi, len(s))
        src_list.append(s); dst_list.append(d); syn_list.append(y)

    if structured:
        # сенсорика -> слой 1 (сходимость, много синапсов на контакт)
        for key in ("vis_l", "vis_r", "olf_good", "olf_bad", "olf_co2",
                    "sugar", "bitter", "water", "jo", "bristle", "hot", "cold"):
            wire(groups[key], l1, 22, 16, 46)
        # слой 1 -> слой 2
        wire(l1, l2, 20, 16, 46)
        # слой 2 -> нисходящие; разные подпучки к разным командам
        half = len(l2) // 2
        wire(l2[:half], groups["turn_left"], 2, 10, 34)
        wire(l2[half:], groups["turn_right"], 2, 10, 34)
        wire(l2, groups["forward"], 1, 10, 30)
        wire(l2[:half // 2], groups["escape"], 1, 8, 24)
        wire(l2, groups["groom"], 1, 6, 20)
        wire(l2, groups["speed"], 1, 6, 20)
        # прямые «командные» ветки, как у настоящих коротких путей
        wire(groups["vis_l"], groups["turn_left"], 1, 14, 40)
        wire(groups["vis_r"], groups["turn_right"], 1, 14, 40)
        # оптический поток тянет за гейт передней ходьбы
        wire(np.concatenate([groups["vis_l"], groups["vis_r"]]),
             groups["forward"], 1, 8, 24)
        wire(groups["olf_good"], groups["forward"], 3, 30, 80)
        wire(groups["sugar"], groups["feed"], 4, 30, 80)
        wire(groups["bitter"], groups["backward"], 4, 30, 80)
        wire(groups["olf_bad"], groups["escape"], 3, 30, 80)
        wire(groups["bristle"], groups["escape"], 2, 20, 60)
        wire(groups["jo"], groups["threat"], 2, 20, 60)
        wire(groups["hot"], groups["escape"], 3, 30, 80)
        # Ноцицепция -> гигантский нейрон: та самая дуга, по которой муха
        # шарахается от удара. Сильнее прочих входов, как у DNp01.
        wire(groups["nocic"], groups["escape"], 5, 60, 140)
        wire(groups["nocic"], groups["backward"], 2, 20, 60)

    # фоновая рекуррентная сеть с тяжёлым хвостом степеней
    deg = np.clip(rng.lognormal(np.log(avg_degree) - 0.5, 1.0, n), 1, 4000).astype(np.int64)
    nnz_bg = int(deg.sum())
    src_list.append(np.repeat(np.arange(n), deg))
    dst_list.append(rng.integers(0, n, size=nnz_bg))
    syn_list.append(np.clip(rng.lognormal(1.6, 0.9, nnz_bg), 5, 500))

    pre = np.concatenate(src_list).astype(np.int64)
    post = np.concatenate(dst_list).astype(np.int64)
    syn = np.concatenate(syn_list)

    w = (syn.astype(np.float32) * signs[pre]
         * np.float32(params.w_synapse) * np.float32(params.gain))
    nz = w != 0
    pre, post, w = pre[nz], post[nz], w[nz]

    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])

    return Connectome(
        indptr=indptr, indices=post.astype(np.int32), weights=w.astype(np.float32),
        root_ids=np.arange(n, dtype=np.uint64) + 720575940000000000,
        cell_type=cell_type, hb_type=cell_type.copy(), klass=klass,
        super_class=super_class, side=side, nt=nt.astype("<U16"), xyz=xyz,
        meta={"source": "synthetic" + ("-structured" if structured else "-random"),
              "seed": seed, "gain": params.gain},
    )
