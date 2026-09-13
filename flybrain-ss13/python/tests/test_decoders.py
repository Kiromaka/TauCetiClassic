"""Декодер моторных команд: приоритеты, рулёжка, базовые линии."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flybrain.decoders import EAST, NORTH, SOUTH, WEST, MotorDecoder  # noqa: E402

KEYS = ("escape", "forward", "backward", "turn_left", "turn_right",
        "groom", "threat", "feed", "speed")


def _warm(dec, n=80, seed=0, over=None):
    rng = np.random.default_rng(seed)
    a = None
    for _ in range(n):
        rates = {k: max(0.0, rng.normal(5, 1)) for k in KEYS}
        if over:
            rates.update(over)
        a = dec.decode(rates, {"turn_left": 2, "turn_right": 2})
    return a


def test_baseline_makes_flat_input_boring():
    dec = MotorDecoder()
    a = _warm(dec)
    assert a.act is None and a.dbg["state"] in ("idle", "forward")


def test_escape_beats_everything():
    dec = MotorDecoder()
    _warm(dec)
    heading_before = dec.heading
    a = dec.decode({**{k: 5.0 for k in KEYS}, "escape": 400.0, "forward": 400.0},
                   {"turn_left": 1, "turn_right": 1})
    assert a.act == "escape" and a.move == 1 and a.run == 1
    # гигантский нейрон разворачивает муху прочь
    assert abs(((dec.heading - heading_before) % 360) - 180) < 60


def test_escape_has_refractory_run():
    """После срабатывания гигантского нейрона муха убегает несколько тиков,
    а потом успокаивается.

    Всплеск виден всё окно накопления (150 мс мозгового времени), поэтому
    состояние "escape" держится не один тик, а пока всплеск в окне; дальше
    идёт инерционный пробег и только потом остановка."""
    dec = MotorDecoder()
    _warm(dec)
    dec.decode({**{k: 5.0 for k in KEYS}, "escape": 400.0}, None)
    seq = [dec.decode({k: 5.0 for k in KEYS}, None) for _ in range(8)]
    states = [a.dbg["state"] for a in seq]
    assert states[0] in ("escape", "escape-run"), states
    assert all(a.move == 1 for a in seq[:2]), states
    assert "escape-run" in states, states
    assert states[-1] not in ("escape", "escape-run"), states


def test_backward_from_mdn():
    dec = MotorDecoder()
    _warm(dec)
    a = dec.decode({**{k: 5.0 for k in KEYS}, "backward": 200.0}, None)
    assert a.move == -1 and a.dbg["state"] == "backward"


def test_feeding_and_grooming():
    dec = MotorDecoder()
    _warm(dec)
    assert dec.decode({**{k: 5.0 for k in KEYS}, "feed": 200.0}, None).act == "eat"
    dec2 = MotorDecoder()
    _warm(dec2)
    assert dec2.decode({**{k: 5.0 for k in KEYS}, "groom": 200.0}, None).act == "groom"


def test_turning_follows_left_right_asymmetry():
    dec = MotorDecoder(spontaneous=False)
    _warm(dec, over={"turn_left": 5.0, "turn_right": 5.0})
    h0 = dec.heading
    for _ in range(6):
        dec.decode({**{k: 5.0 for k in KEYS}, "turn_right": 90.0}, None)
    right = (dec.heading - h0) % 360
    dec2 = MotorDecoder(spontaneous=False)
    _warm(dec2, over={"turn_left": 5.0, "turn_right": 5.0})
    h0 = dec2.heading
    for _ in range(6):
        dec2.decode({**{k: 5.0 for k in KEYS}, "turn_left": 90.0}, None)
    left = (h0 - dec2.heading) % 360
    assert right > 20 and left > 20, (right, left)


def test_heading_snaps_to_four_dirs():
    # dir_min_ticks=0: тест телепортирует курс, а выдержка между сменами
    # рассчитана на непрерывное движение — она проверяется отдельно ниже.
    dec = MotorDecoder(spontaneous=False, dir_min_ticks=0)
    seen = set()
    for h, want in ((0, NORTH), (90, EAST), (180, SOUTH), (270, WEST)):
        dec.heading = h
        a = dec.decode({k: 0.0 for k in KEYS}, None)
        seen.add(a.dir)
        assert a.dir == want, (h, a.dir, want)
    assert seen == {NORTH, SOUTH, EAST, WEST}


def test_spontaneous_noise_moves_the_fly():
    """Без внятных стимулов муха всё равно должна бродить,
    опираясь на собственный шум сети."""
    dec = MotorDecoder(spontaneous=True)
    rng = np.random.default_rng(5)
    h0 = dec.heading
    for _ in range(60):
        dec.decode({k: 5.0 for k in KEYS},
                   {"turn_left": int(rng.integers(0, 6)),
                    "turn_right": int(rng.integers(0, 6))})
    assert dec.heading != h0


def test_diagnosis_names_the_stage():
    """Одна и та же надпись «моторика молчит» бывает по трём разным
    причинам, и лечатся они по-разному. Диагноз должен их различать."""
    import numpy as np
    from flybrain.agent import BrainPool
    from flybrain.config import Config, PopulationSpec
    from flybrain.connectome import make_synthetic

    cx = make_synthetic(4000, 30, seed=5)

    # 1) мозг не спайкует: веса задавлены в ноль
    cfg = Config()
    cfg.runtime.weight_scale = 0.0
    cfg.runtime.background_hz = 0.0
    cfg.runtime.input_drive_mv = 0.0
    cfg.runtime.brain_ms_per_tick = 20.0
    ag = BrainPool(cfg, cx).get("d1")
    ag.step({"view_w": 15, "view_h": 15})
    d = ag.diagnose()
    assert d["cause"] == "мозг не спайкует", d
    assert "calibrate.py --sweep" in d["fix"]
    assert d["weight_scale"] == 0.0

    # 2) популяции не нашлись
    cfg2 = Config()
    cfg2.runtime.brain_ms_per_tick = 20.0
    for k in list(cfg2.motor):
        cfg2.motor[k] = PopulationSpec(match_type=r"ЗАВЕДОМО_НЕТ_ТАКОГО",
                                       match_super=r"ЗАВЕДОМО_НЕТ_ТАКОГО")
    ag2 = BrainPool(cfg2, cx).get("d2")
    ag2.step({"view_w": 15, "view_h": 15})
    d2 = ag2.diagnose()
    assert d2["cause"] == "популяции не нашлись", d2
    assert len(d2["empty"]) >= 4, d2["empty"]


def test_diagnosis_is_quiet_when_brain_drives():
    from flybrain.agent import BrainPool
    from flybrain.config import Config
    from flybrain.connectome import make_synthetic
    cfg = Config()
    cfg.runtime.brain_ms_per_tick = 30.0
    ag = BrainPool(cfg, make_synthetic(4000, 30, seed=7)).get("ok")
    for _ in range(4):
        ag.step({"view_w": 15, "view_h": 15, "light": "5" * 225})
    d = ag.diagnose()
    if d["active"]:
        assert d["cause"] == "", d


def test_dir_does_not_flip_on_a_boundary():
    """Курс, зависший у границы секторов, не должен перекидывать dir каждый
    тик. Со стороны это и выглядит как "муха вертится вокруг своей оси"."""
    dec = MotorDecoder(spontaneous=False, fallback=False)
    dirs = []
    for i in range(40):
        dec.heading = 45.0 + (1.5 if i % 2 else -1.5)     # дрожь у границы
        dirs.append(dec.decode({k: 0.0 for k in KEYS}, None).dir)
    flips = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
    assert flips <= 1, (flips, dirs[:12])


def test_dir_still_follows_a_real_turn():
    """Гистерезис не должен запирать муху: настоящий поворот проходит."""
    dec = MotorDecoder(spontaneous=False, fallback=False)
    dec.decode({k: 0.0 for k in KEYS}, None)
    seen = []
    for i in range(24):
        dec.heading = (i * 30.0) % 360.0
        seen.append(dec.decode({k: 0.0 for k in KEYS}, None).dir)
    assert len(set(seen)) == 4, seen


def test_steady_channel_does_not_pin_the_ladder():
    """Ровный канал раньше набирал z под +8 и намертво занимал верх
    лестницы: муха до конца раунда пятилась назад."""
    import random
    dec = MotorDecoder(spontaneous=False, fallback=False, common_mode=False)
    random.seed(4)
    states = []
    for t in range(600):
        rates = {k: 1.0 for k in KEYS}
        rates["backward"] = 34.7 + random.gauss(0, 0.3)
        rates["groom"] = 119.6 + random.gauss(0, 0.8)
        spikes = {k: int(rates[k] * 3) for k in rates}
        a = dec.decode(rates, spikes)
        if t > 200:
            states.append(a.dbg["state"])
    back = states.count("backward") / len(states)
    assert back < 0.25, (back, dec.snapshot()["z"])


def test_big_change_still_fires():
    """Пол дисперсии не должен сделать декодер глухим."""
    import random
    dec = MotorDecoder(spontaneous=False, fallback=False, common_mode=False)
    random.seed(5)
    for t in range(300):
        rates = {k: 1.0 for k in KEYS}
        rates["backward"] = 30.0 + random.gauss(0, 0.4)
        dec.decode(rates, {k: int(rates[k] * 3) for k in rates})
    rates = {k: 1.0 for k in KEYS}
    rates["backward"] = 90.0
    a = dec.decode(rates, {k: int(rates[k] * 3) for k in rates})
    assert a.dbg["state"] == "backward", (a.dbg["state"], dec.snapshot()["z"])


def test_constant_bias_does_not_spin_the_fly():
    """Главная причина верчения вокруг своей оси: разность z двух рулевых
    каналов не нулевая в среднем. Частоты считаются из горстки спайков за
    тик, распределение счёта скошено, и у более "взрывного" канала
    большинство тиков оказывается ниже собственного среднего. Постоянная
    добавка упирает рулёжку в ограничитель, и курс едет по кругу."""
    import random
    dec = MotorDecoder(spontaneous=False, fallback=False)
    random.seed(11)
    heads = []
    for t in range(400):
        rates = {k: 1.0 for k in KEYS}
        # правый канал систематически "взрывнее" левого
        rates["turn_left"] = 20.0 + random.gauss(0, 3)
        rates["turn_right"] = 20.0 + random.gauss(0, 3) + (9.0 if random.random() < 0.15 else 0)
        rates["dn_left"] = 18.0 + random.gauss(0, 2)
        rates["dn_right"] = 22.0 + random.gauss(0, 2)
        spikes = {k: max(1, int(rates[k] * 0.3)) for k in rates}
        a = dec.decode(rates, spikes)
        if t > 150:
            heads.append(a.dbg["heading"])
    step = [abs(((b - x + 180) % 360) - 180) for x, b in zip(heads, heads[1:])]
    mean = sum(step) / len(step)
    assert mean < 12.0, mean          # раньше упиралось в ограничитель ~37
    # и курс не должен наматывать полный оборот за десяток тиков
    total = sum(((b - x + 180) % 360) - 180 for x, b in zip(heads, heads[1:]))
    assert abs(total) < 360 * 1.5, total


def test_a_real_asymmetry_still_turns_the_fly():
    """Снятие перекоса не должно сделать муху неповоротливой: устойчивое
    ИЗМЕНЕНИЕ баланса обязано поворачивать."""
    import random
    dec = MotorDecoder(spontaneous=False, fallback=False)
    random.seed(12)

    def feed(right_extra, n):
        h0 = dec.heading
        for _ in range(n):
            rates = {k: 1.0 for k in KEYS}
            rates["turn_left"] = 20.0 + random.gauss(0, 2)
            rates["turn_right"] = 20.0 + right_extra + random.gauss(0, 2)
            rates["dn_left"] = 20.0
            rates["dn_right"] = 20.0 + right_extra
            dec.decode(rates, {k: max(1, int(rates[k] * 0.6)) for k in rates})
        return ((dec.heading - h0 + 180) % 360) - 180

    feed(0.0, 200)                       # снимаем перекос на симметрии
    turned = feed(25.0, 12)              # резкий крен вправо
    assert turned > 25, turned


def test_evidence_window_lifts_small_populations():
    """На коннектоме пользователя ноцицепция поднимала backward всего на
    +12.5 Гц при четырёх нейронах: пуассоновский шум за один тик около
    12 Гц, z выходил 1.0 при пороге 1.26, и муха на удар не реагировала.
    Окно накопления это чинит, не трогая порог."""
    import random

    def probe(win):
        dec = MotorDecoder(spontaneous=False, fallback=False,
                           common_mode=False, evidence_ticks=win)
        random.seed(3)

        def tick(extra=0.0):
            rates = {k: 1.0 for k in KEYS}
            rates["backward"] = 16.2 + extra + random.gauss(0, 1.5)
            # 4 нейрона за 50 мс -> 0.2 нейрон-секунды
            sp = {k: max(0, int(rates[k] * 0.2)) for k in rates}
            return dec.decode(rates, sp)

        for _ in range(300):
            tick()
        out = [tick(12.5) for _ in range(8)]
        return max(dec._last_z["backward"] for _ in [0]), [a.dbg["state"] for a in out]

    z1, st1 = probe(1)
    z3, st3 = probe(3)
    assert "backward" not in st1, (z1, st1)     # так было
    assert "backward" in st3, (z3, st3)         # так стало
