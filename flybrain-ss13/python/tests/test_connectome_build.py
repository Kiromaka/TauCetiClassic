"""Сборка коннектома из CSV: терпимость к разным схемам выгрузки.

Состав файлов FlyWire меняется от снапшота к снапшоту. В свежих выгрузках
в classification.csv нет колонки cell_type — типы уехали в
consolidated_cell_types.csv. Загрузчик обязан это переживать.
"""
import csv
import gzip
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flybrain.config import (default_motor_specs,                     # noqa: E402
                             default_sensory_specs)
from flybrain.connectome import (build, columns_of,                   # noqa: E402
                                 find_type_sources)
from flybrain.populations import resolve                              # noqa: E402

ROOT0 = 720575940000000000
N = 600


def _write(path, header, rows):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _type_of(i):
    if i < 60:    return "L1", "visual", "optic"
    if i < 90:    return "ORN_DM1", "olfactory", "sensory"
    if i < 110:   return "Gr5a", "gustatory", "sensory"
    if i < 114:   return "DNp09", "descending", "descending"
    if i < 118:   return "MDN", "descending", "descending"
    if i < 126:   return "DNa01", "descending", "descending"
    if i < 130:   return "DNp01", "descending", "descending"
    return "", "", "central"


def _make(dirpath, cell_type_in_classification: bool, with_nt: bool = True):
    roots = [ROOT0 + i for i in range(N)]
    clas_header = ["root_id", "flow", "super_class", "class", "sub_class",
                   "hemilineage", "side", "nerve"]
    if cell_type_in_classification:
        clas_header.insert(5, "cell_type")
    rows = []
    for i, r in enumerate(roots):
        t, cls, sup = _type_of(i)
        row = [r, "intrinsic", sup, cls, "", "", ("left" if i % 2 else "right"), ""]
        if cell_type_in_classification:
            row.insert(5, t)
        rows.append(row)
    _write(os.path.join(dirpath, "classification.csv.gz"), clas_header, rows)

    if not cell_type_in_classification:
        _write(os.path.join(dirpath, "consolidated_cell_types.csv.gz"),
               ["root_id", "primary_type", "additional_type(s)"],
               [[r, _type_of(i)[0], ""] for i, r in enumerate(roots)
                if _type_of(i)[0]])

    rng = np.random.default_rng(0)
    conn_header = ["pre_root_id", "post_root_id", "neuropil", "syn_count"]
    if with_nt:
        conn_header.append("nt_type")
    crows = []
    for _ in range(9000):
        a, b = int(rng.integers(0, N)), int(rng.integers(0, N))
        row = [roots[a], roots[b], "ME_L", int(rng.integers(5, 120))]
        if with_nt:
            row.append(["ACH", "ACH", "GABA", "GLUT"][int(rng.integers(0, 4))])
        crows.append(row)
    _write(os.path.join(dirpath, "connections.csv.gz"), conn_header, crows)
    return (os.path.join(dirpath, "connections.csv.gz"),
            os.path.join(dirpath, "classification.csv.gz"))


def _check_populations(cx):
    pops = resolve(cx, default_sensory_specs(), default_motor_specs())
    for name in ("forward", "backward", "turn_left", "turn_right", "escape"):
        assert len(pops.motor[name]) > 0, f"{name} не нашлась\n{pops.summary()}"
    assert len(pops.sensory["vis_left"]) > 0
    assert len(pops.sensory["gust_sugar"]) > 0


def test_types_in_separate_file():
    """Схема свежих снапшотов: cell_type нет в classification."""
    d = tempfile.mkdtemp()
    try:
        conn, clas = _make(d, cell_type_in_classification=False)
        assert "cell_type" not in columns_of(clas)
        srcs = find_type_sources(d)
        assert any("consolidated" in os.path.basename(p) for p in srcs), srcs
        cx = build(conn, clas, type_csvs=srcs, verbose=False)
        assert (cx.cell_type == "DNp09").sum() == 4
        _check_populations(cx)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_types_inline_still_work():
    """Старая схема (типы прямо в classification) обязана работать по-прежнему."""
    d = tempfile.mkdtemp()
    try:
        conn, clas = _make(d, cell_type_in_classification=True)
        cx = build(conn, clas, type_csvs=find_type_sources(d), verbose=False)
        assert (cx.cell_type == "MDN").sum() == 4
        _check_populations(cx)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_missing_types_gives_actionable_error():
    d = tempfile.mkdtemp()
    try:
        conn, clas = _make(d, cell_type_in_classification=False)
        os.unlink(os.path.join(d, "consolidated_cell_types.csv.gz"))
        try:
            build(conn, clas, type_csvs=find_type_sources(d), verbose=False)
            raise AssertionError("сборка без типов должна падать")
        except RuntimeError as exc:
            assert "consolidated_cell_types" in str(exc)
            assert "DNp09" in str(exc)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_connections_without_nt_column():
    """Если в связях нет nt_type, медиатор берётся из аннотаций."""
    d = tempfile.mkdtemp()
    try:
        roots = [ROOT0 + i for i in range(N)]
        conn, clas = _make(d, cell_type_in_classification=False, with_nt=False)
        # добавляем медиатор отдельным файлом
        _write(os.path.join(d, "neurons.csv.gz"), ["root_id", "nt_type"],
               [[r, "ACH" if i % 3 else "GABA"] for i, r in enumerate(roots)])
        assert "nt_type" not in columns_of(conn)
        cx = build(conn, clas, type_csvs=find_type_sources(d) +
                   [os.path.join(d, "neurons.csv.gz")], verbose=False)
        assert cx.nnz > 0, "все связи потерялись без медиатора"
        assert (cx.weights < 0).any() and (cx.weights > 0).any()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_missing_side_column_is_flagged():
    """Без стороны тела муха не сможет рулить — это должно быть видно сразу."""
    d = tempfile.mkdtemp()
    try:
        roots = [ROOT0 + i for i in range(N)]
        _write(os.path.join(d, "classification.csv.gz"),
               ["root_id", "super_class", "class"],
               [[r, _type_of(i)[2], _type_of(i)[1]] for i, r in enumerate(roots)])
        _write(os.path.join(d, "consolidated_cell_types.csv.gz"),
               ["root_id", "primary_type"],
               [[r, _type_of(i)[0]] for i, r in enumerate(roots) if _type_of(i)[0]])
        _write(os.path.join(d, "connections.csv.gz"),
               ["pre_root_id", "post_root_id", "syn_count", "nt_type"],
               [[roots[i % N], roots[(i * 7) % N], 30, "ACH"] for i in range(4000)])
        cx = build(os.path.join(d, "connections.csv.gz"),
                   os.path.join(d, "classification.csv.gz"),
                   type_csvs=find_type_sources(d), verbose=False)
        pops = resolve(cx, default_sensory_specs(), default_motor_specs())
        assert any("сторон" in line for line in pops.report), pops.report
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_extra_unknown_columns_are_ignored():
    d = tempfile.mkdtemp()
    try:
        roots = [ROOT0 + i for i in range(N)]
        _write(os.path.join(d, "classification.csv.gz"),
               ["root_id", "super_class", "class", "side", "какая-то_новая_колонка"],
               [[r, _type_of(i)[2], _type_of(i)[1],
                 ("left" if i % 2 else "right"), "мусор"]
                for i, r in enumerate(roots)])
        _write(os.path.join(d, "consolidated_cell_types.csv.gz"),
               ["root_id", "primary_type"],
               [[r, _type_of(i)[0]] for i, r in enumerate(roots) if _type_of(i)[0]])
        _write(os.path.join(d, "connections.csv.gz"),
               ["pre_root_id", "post_root_id", "syn_count", "nt_type"],
               [[roots[i % N], roots[(i * 7) % N], 30, "ACH"] for i in range(5000)])
        cx = build(os.path.join(d, "connections.csv.gz"),
                   os.path.join(d, "classification.csv.gz"),
                   type_csvs=find_type_sources(d), verbose=False)
        _check_populations(cx)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_photoreceptors_without_synapses_are_skipped():
    """Ловушка настоящего FAFB: типы R1-R8 в аннотациях есть, а синапсов нет.

    Если выбрать такой слой «сетчаткой», накачка упрётся в тупик и ни один
    нисходящий нейрон не шевельнётся. Резолвер обязан это заметить и взять
    следующий слой.
    """
    from flybrain.connectome import make_synthetic

    cx = make_synthetic(8000, 60, seed=13)
    # берём нейроны с самой бедной исходящей связностью и объявляем их R1-6
    out_deg = np.diff(cx.indptr)
    poor = np.argsort(out_deg)[:400]
    cx.cell_type[poor] = "R1-6"
    cx.klass[poor] = "visual"
    cx.super_class[poor] = "optic"
    cx.side[poor] = np.where(np.arange(len(poor)) % 2, "left", "right")

    pops = resolve(cx, default_sensory_specs(), default_motor_specs())
    report = "\n".join(pops.report)
    assert "R[1-8]" not in report.split("зрительный вход:")[1].split("\n")[0], report
    assert any("пропущен слой" in line for line in pops.report), report
    # и выбранный слой должен реально иметь исходящие связи
    chosen = pops.sensory["vis_left"].idx
    assert np.median(np.diff(cx.indptr)[chosen]) > 1


def test_collapsed_populations_are_reported():
    """Если сладкое и горькое схлопнулись в один набор — сказать об этом."""
    from flybrain.connectome import make_synthetic
    from flybrain.config import PopulationSpec

    cx = make_synthetic(6000, 40, seed=14)
    sens = default_sensory_specs()
    # обе популяции ищут несуществующий тип и откатятся на весь класс
    sens["gust_sugar"] = PopulationSpec(match_type=r"НЕТ_ТАКОГО_1",
                                        match_class=r"gustatory")
    sens["gust_bitter"] = PopulationSpec(match_type=r"НЕТ_ТАКОГО_2",
                                         match_class=r"gustatory")
    pops = resolve(cx, sens, default_motor_specs())
    assert any("одинаковый набор нейронов" in line and "gust_sugar" in line
               for line in pops.report), pops.report
