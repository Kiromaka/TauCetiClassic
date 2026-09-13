"""Избирательность каналов: общая мода, популяционная рулёжка, разделение.

На настоящем коннектоме первая же калибровка показала, что один канал
(turn_right) отвечает почти на любой стимул. Это не избирательность, а
общий уровень активности, протекающий в самый чувствительный канал.
"""
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flybrain.analysis import orient_clusters, split_population   # noqa: E402
from flybrain.config import Config                                 # noqa: E402
from flybrain.connectome import make_synthetic                     # noqa: E402
from flybrain.decoders import COMMANDS, MotorDecoder               # noqa: E402
from flybrain.populations import resolve                           # noqa: E402

BASE = {"escape": 8.0, "forward": 6.0, "backward": 18.0, "turn_left": 64.0,
        "turn_right": 12.0, "groom": 101.0, "threat": 0.2, "feed": 0.1,
        "speed": 15.0}


def _warm(dec, ticks=140, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(ticks):
        dec.decode({k: max(0.0, v * rng.normal(1, 0.05)) for k, v in BASE.items()})


def test_common_mode_is_rejected():
    """Равномерный подъём всех каналов не должен выглядеть как сигнал."""
    off = MotorDecoder(common_mode=False, pop_steering=0.0)
    on = MotorDecoder(common_mode=True, pop_steering=0.0)
    _warm(off)
    _warm(on)
    lift = {k: v * 1.25 for k, v in BASE.items()}
    for _ in range(6):
        a = off.decode(lift)
        b = on.decode(lift)
    worst_off = max(abs(v) for v in a.dbg["z"].values())
    worst_on = max(abs(v) for v in b.dbg["z"].values())
    assert worst_on < worst_off * 0.7, (worst_on, worst_off)


def test_specific_signal_survives_common_mode():
    """А настоящий, свой для канала сигнал должен пройти насквозь."""
    dec = MotorDecoder(common_mode=True, pop_steering=0.0)
    _warm(dec)
    for _ in range(6):
        a = dec.decode({**BASE, "turn_right": BASE["turn_right"] * 2})
    z = a.dbg["z"]
    assert z["turn_right"] > 2.0, z
    others = [abs(v) for k, v in z.items() if k != "turn_right"]
    assert max(others) < 1.0, z


def test_population_steering_adds_signal():
    """Асимметрия всей нисходящей популяции должна поворачивать муху."""
    dec = MotorDecoder(pop_steering=1.0)
    flat = {**BASE, "dn_left": 5.0, "dn_right": 5.0}
    for _ in range(60):
        dec.decode(flat)
    h0 = dec.heading
    for _ in range(6):
        dec.decode({**BASE, "dn_left": 5.0, "dn_right": 40.0})
    assert (dec.heading - h0) % 360 > 15, dec.heading
    assert dec.source == "мозг"


def test_tiny_turn_population_is_widened():
    """Три нейрона на команду — шумно; шаблон должен расшириться с отчётом."""
    cfg = Config.load()
    cx = make_synthetic(9000, 40, seed=17, params=cfg.model)
    # оставляем всего по паре DNa01, как в настоящей выгрузке
    keep = np.flatnonzero(cx.cell_type == "DNa02")
    cx.cell_type[keep[2:]] = "DNa05"          # тип есть, но не из основного шаблона
    pops = resolve(cx, cfg.sensory, cfg.motor)
    note = pops.motor["turn_left"].note + pops.motor["turn_right"].note
    assert "расширено" in note or len(pops.motor["turn_left"]) >= 6, note


def test_unlabelled_class_is_split_by_connectivity():
    """Вкусовой класс без подтипов делится по тому, куда он проецируется."""
    cfg = Config.load()
    cx = make_synthetic(16000, 50, seed=23, params=cfg.model)
    rng = np.random.default_rng(1)
    gust = np.flatnonzero(cx.klass == "gustatory")
    cx.cell_type[gust] = ""
    cx.hb_type[gust] = ""
    half = len(gust) // 2
    sweet, bitter = gust[:half], gust[half:]
    feed = np.flatnonzero(cx.cell_type == "MN9")
    back = np.flatnonzero(cx.cell_type == "MDN")

    pre, post, w = [], [], []
    for src, dst in ((sweet, feed), (bitter, back)):
        for a in src:
            for _ in range(6):
                pre.append(int(a)); post.append(int(rng.choice(dst))); w.append(9.0)
    owner = np.repeat(np.arange(cx.n), np.diff(cx.indptr))
    pre = np.concatenate([owner, np.array(pre, np.int64)])
    post = np.concatenate([cx.indices.astype(np.int64), np.array(post, np.int64)])
    w = np.concatenate([cx.weights, np.array(w, np.float32)])
    o = np.argsort(pre, kind="stable")
    pre, post, w = pre[o], post[o], w[o]
    cx.indptr = np.zeros(cx.n + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=cx.n), out=cx.indptr[1:])
    cx.indices, cx.weights = post.astype(np.int32), w.astype(np.float32)

    pops = resolve(cx, cfg.sensory, cfg.motor)
    assert any("нет подтипов" in line for line in pops.report), pops.report
    assert any("ВНИМАНИЕ" in line for line in pops.report), "ярлыки надо оговаривать"

    got_sweet = set(pops.sensory["gust_sugar"].idx.tolist())
    got_bitter = set(pops.sensory["gust_bitter"].idx.tolist())
    ts, tb = set(sweet.tolist()), set(bitter.tolist())
    assert len(got_sweet) and len(got_bitter)
    assert not (got_sweet & got_bitter), "кластеры пересеклись"
    purity_sweet = len(got_sweet & ts) / len(got_sweet)
    purity_bitter = len(got_bitter & tb) / len(got_bitter)
    assert purity_sweet > 0.8, purity_sweet
    assert purity_bitter > 0.8, purity_bitter


def test_split_is_disableable():
    cfg = Config.load()
    cx = make_synthetic(9000, 40, seed=5, params=cfg.model)
    gust = np.flatnonzero(cx.klass == "gustatory")
    cx.cell_type[gust] = ""
    cx.hb_type[gust] = ""
    pops = resolve(cx, cfg.sensory, cfg.motor, auto_split=False)
    assert any("не различает" in line for line in pops.report), pops.report


def test_shuffled_connectome_keeps_statistics_but_loses_structure():
    """Нулевая модель должна быть той же физики, но без путей."""
    from flybrain.analysis import shuffle_connectome
    cx = make_synthetic(5000, 40, seed=3)
    sh = shuffle_connectome(cx, seed=1)
    assert np.array_equal(np.diff(cx.indptr), np.diff(sh.indptr))
    assert np.array_equal(np.sort(cx.weights), np.sort(sh.weights))
    assert not np.array_equal(cx.indices, sh.indices)
    # аннотации должны сохраниться, иначе популяции не разрешатся
    assert np.array_equal(cx.cell_type, sh.cell_type)


def test_arena_speaks_the_same_protocol_as_dm():
    """Арена для проб должна кодировать поле зрения так же, как DM."""
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__))))
    from arena import Arena, NORTH, SOUTH, EAST, WEST
    from test_protocol import dm_cell_index, dm_pack

    a = Arena(seed=1)
    for facing in (NORTH, SOUTH, EAST, WEST):
        a.fdir = facing
        for dx in (-7, -3, 0, 4, 7):
            for dy in (-7, 0, 5, 7):
                assert a.to_fly(dx, dy) == dm_cell_index(dx, dy, facing)
    assert dm_pack([0.0, 0.5, 1.0]) == "059"
    p = a.percept(0)
    assert len(p["light"]) == 225 and len(p["solid"]) == 225
    for k in ("threat_fx", "threat_fy", "threat_near", "nutrition",
              "food_fx", "food_fy", "food_near", "food_underfoot", "food_touch"):
        assert k in p, k

    # Поворот в систему координат мухи должен совпадать с DM-процедурой
    # flybrain_body_frame: пеленг еды и угрозы считается именно им.
    def dm_body_frame(dx, dy, facing):
        if facing == SOUTH:
            return (-dx, -dy)
        if facing == EAST:
            return (-dy, dx)
        if facing == WEST:
            return (dy, -dx)
        return (dx, dy)

    for facing in (NORTH, SOUTH, EAST, WEST):
        a.fdir = facing
        for dx, dy in ((3, 0), (0, -4), (-2, 5), (6, 6)):
            assert a.fly_frame(dx, dy) == dm_body_frame(dx, dy, facing), (
                facing, dx, dy)


def test_widen_cannot_swallow_the_whole_motor_pool():
    """Расширение шаблоном '.' превращало feed во всю нисходящую популяцию,
    и муха тянулась к еде на любой раздражитель."""
    from flybrain.config import PopulationSpec
    cfg = Config.load()
    cx = make_synthetic(12000, 50, seed=8, params=cfg.model)
    motor = dict(cfg.motor)
    motor["greedy"] = PopulationSpec(match_type=r"НЕТ_ТАКОГО",
                                     match_super=r"descending|motor",
                                     widen=r".", min_size=1)
    pops = resolve(cx, cfg.sensory, motor)
    dn = len(pops.motor["dn_all"])
    assert len(pops.motor["greedy"]) <= max(8, dn // 4), (
        len(pops.motor["greedy"]), dn, pops.motor["greedy"].note)


def test_effect_size_is_paired():
    """Раньше настоящий мозг гонялся на аренах 0..N, а перемешанный на
    100..100+N — сравнивались разные комнаты, и разброс арены маскировал
    разницу мозгов."""
    import importlib.util
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "bt", os.path.join(here, "..", "scripts", "behaviour_test.py"))
    bt = importlib.util.module_from_spec(spec)
    sys.modules["bt"] = bt
    spec.loader.exec_module(bt)

    # Общий шум комнаты одинаков в обеих ветках и должен вычитаться.
    room = [0.0, 9.0, -7.0, 4.0, 12.0, -3.0]
    real = [x + 1.0 for x in room]
    null = [x for x in room]
    d = bt.cohen_d(real, null)
    p = bt.permutation_p(real, null, iters=4000, seed=2)
    assert d > 5.0, d            # разброс разностей нулевой -> упираемся в пол
    assert p < 0.05, p

    # А когда разницы нет — теста быть не должно
    d0 = bt.cohen_d(room, list(room))
    p0 = bt.permutation_p(room, list(room), iters=4000, seed=2)
    assert abs(d0) < 1e-6, d0
    assert p0 > 0.9, p0

    # Непарные длины не должны падать
    assert bt.cohen_d([1.0, 2.0], [1.0]) == 0.0
    assert bt.permutation_p([1.0, 2.0], [1.0]) == 1.0


def test_tiny_generic_widen_is_rejected():
    """На выгрузке пользователя generic-шаблон дал для feed ТРИ нейрона.
    Такая популяция насыщается от любого входа: в калибровке канал отвечал
    +250 Гц и на сахар, и на воду, и на стену. Честнее оставить канал
    пустым — тогда работает рефлекторная дуга и в dbg видно, что решает не
    коннектом."""
    from flybrain.config import PopulationSpec
    cfg = Config.load()
    cx = make_synthetic(9000, 40, seed=31, params=cfg.model)
    # прячем настоящий пищевой тип, оставляем только три случайных нейрона,
    # которые поймает широкий шаблон
    mn9 = np.flatnonzero(cx.cell_type == "MN9")
    cx.cell_type[mn9] = "MN77"
    cx.hb_type[mn9] = ""
    cx.cell_type[mn9[3:]] = ""
    specs = dict(cfg.motor)
    specs["feed"] = PopulationSpec(match_type=r"^MN9\b",
                                   widen=r"^MN\d+", min_size=1)
    pops = resolve(cx, cfg.sensory, specs)
    assert len(pops.motor["feed"]) == 0, len(pops.motor["feed"])
    note = pops.motor["feed"].note
    assert "НЕ НАЙДЕНО" in note or "отклонено" in note, note


def test_big_enough_widen_still_applies():
    """Отказ от крошечных расширений не должен ломать нормальные."""
    from flybrain.config import PopulationSpec
    cfg = Config.load()
    cx = make_synthetic(9000, 40, seed=31, params=cfg.model)
    specs = dict(cfg.motor)
    specs["feed"] = PopulationSpec(match_type=r"ЗАВЕДОМО_НЕТ",
                                   widen=r"^MN\d+", min_size=1)
    pops = resolve(cx, cfg.sensory, specs)
    assert len(pops.motor["feed"]) >= 4, len(pops.motor["feed"])
    assert "расширено" in pops.motor["feed"].note


def test_channel_search_finds_the_real_feeding_population():
    """Поиск по графу должен находить именно ту популяцию, которую сахар
    тянет специфично, и отвергать нисходящие, которые тянет всё."""
    import importlib.util
    import os as _os
    import sys as _sys
    here = _os.path.dirname(_os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "fch", _os.path.join(here, "..", "scripts", "find_channel.py"))
    fch = importlib.util.module_from_spec(spec)
    _sys.modules["fch"] = fch
    spec.loader.exec_module(fch)

    cfg = Config.load()
    cx = make_synthetic(12000, 108, seed=5, params=cfg.model)
    pops = resolve(cx, cfg.sensory, cfg.motor)
    inf = fch.specificity(cx, pops, "feed", hops=3)
    mine = inf["gust_sugar"]
    others = np.zeros_like(mine)
    for name, v in inf.items():
        if name != "gust_sugar":
            np.maximum(others, v, out=others)

    mn9 = np.flatnonzero(cx.cell_type == "MN9")
    dn = np.flatnonzero(np.char.startswith(cx.cell_type.astype(str), "DNp"))
    assert len(mn9) and len(dn)
    # сахар должен тянуть пищевые моторные нейроны много сильнее, чем
    # нисходящие «побег/ходьба»
    assert mine[mn9].mean() > others[mn9].mean() * 3, (
        mine[mn9].mean(), others[mn9].mean())
    assert mine[dn].mean() < others[dn].mean(), (
        mine[dn].mean(), others[dn].mean())
