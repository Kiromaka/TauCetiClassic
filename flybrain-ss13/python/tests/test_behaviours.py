"""Четыре поведения, добавленные по биологии дрозофилы.

Всё, что здесь проверяется, — реальные и хорошо описанные вещи:
этанол-таксис и три режима опьянения, жажда и канал ppk28,
сомато­топия щетинок с иерархией груминга, отрицательный геотаксис.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arena import Arena                                       # noqa: E402
from flybrain.agent import BrainPool                           # noqa: E402
from flybrain.config import Config                             # noqa: E402
from flybrain.connectome import make_synthetic                 # noqa: E402
from flybrain.decoders import COMMANDS, AUX, MotorDecoder      # noqa: E402
from flybrain.encoders import Percept, SensoryEncoder          # noqa: E402

KEYS = list(COMMANDS) + list(AUX)
_CX = None


def _pool(seed=7, n=9000):
    global _CX
    cfg = Config()
    cfg.runtime.brain_ms_per_tick = 25.0
    if _CX is None:
        _CX = make_synthetic(n, 40, seed=seed, params=cfg.model)
    return BrainPool(cfg, _CX)


def _quiet(dec, ticks=60):
    for _ in range(ticks):
        dec.decode({k: 5.0 for k in KEYS}, None)


# ---------------------------------------------------------------------------
# Спирт
# ---------------------------------------------------------------------------
def test_three_regimes_of_intoxication():
    """У дрозофилы доза этанола даёт три разных режима, и это один из самых
    воспроизводимых эффектов в этой области: малая — гиперактивность и
    расторможенность, средняя — потеря координации, большая — седация."""
    seen = {}
    for drunk in (0.15, 0.45, 0.9):
        dec = MotorDecoder(spontaneous=False, fallback=False)
        _quiet(dec)
        states = [dec.decode({k: 5.0 for k in KEYS}, None, drunk=drunk).dbg["state"]
                  for _ in range(14)]
        seen[drunk] = states
    assert "навеселе" in seen[0.15], seen[0.15]
    assert "шатается" in seen[0.45], seen[0.45]
    assert all(s == "отключилась" for s in seen[0.9]), seen[0.9]


def test_drunk_fly_staggers_more_than_sober():
    dec_s = MotorDecoder(spontaneous=False, fallback=False)
    dec_d = MotorDecoder(spontaneous=False, fallback=False)
    _quiet(dec_s)
    _quiet(dec_d)

    def wobble(dec, drunk):
        hs = []
        for _ in range(40):
            a = dec.decode({k: 5.0 for k in KEYS}, None, drunk=drunk)
            hs.append(a.dbg["heading"])
        return sum(abs(((b - x + 180) % 360) - 180)
                   for x, b in zip(hs, hs[1:])) / (len(hs) - 1)

    assert wobble(dec_d, 0.5) > wobble(dec_s, 0.0) * 1.5


def test_tolerance_softens_the_same_dose():
    """Толерантность у мухи развивается быстро: та же доза со временем
    валит слабее."""
    pops = _pool().pops
    enc = SensoryEncoder(pops, Config().runtime)
    fresh = enc.effective_drunk(0.8)
    for _ in range(60):
        enc.encode(Percept.from_json({"view_w": 15, "view_h": 15, "drunk": 0.8}))
    seasoned = enc.effective_drunk(0.8)
    assert seasoned < fresh * 0.8, (fresh, seasoned)


def test_drunk_fly_wants_booze_less():
    """Уже пьяная муха тянется к спирту заметно слабее."""
    dec = MotorDecoder(spontaneous=False, fallback=False)
    _quiet(dec)
    sober = dec._pick_goal(
        z={k: 0.0 for k in KEYS}, hunger=0.5, taste=0.0, food_near=0.0,
        food_bearing=None, thirst=0.0, water_near=0.0, water_bearing=None,
        booze_near=0.8, booze_bearing=0.0, drunk=0.0,
        climb_near=0.0, climb_bearing=None, on_high=0.0)
    tipsy = dec._pick_goal(
        z={k: 0.0 for k in KEYS}, hunger=0.5, taste=0.0, food_near=0.0,
        food_bearing=None, thirst=0.0, water_near=0.0, water_bearing=None,
        booze_near=0.8, booze_bearing=0.0, drunk=0.9,
        climb_near=0.0, climb_bearing=None, on_high=0.0)
    assert sober.name == "спирт" and tipsy.name == "спирт"
    assert tipsy.drive < sober.drive * 0.5, (sober.drive, tipsy.drive)


def test_hungry_fly_flies_to_fermentation():
    pool = _pool()
    a = Arena(seed=3, clutter=6)
    a.booze = (a.fx + 5, a.fy)
    ag = pool.get("booze")
    for _ in range(30):
        ag.step(a.percept(-1))
    for t in range(90):
        a.apply(ag.step(a.percept(t)))
    assert a.drinks >= 1, a.drinks
    assert a.drunk > 0.1, a.drunk


# ---------------------------------------------------------------------------
# Жажда и вода
# ---------------------------------------------------------------------------
def test_water_channel_opens_only_when_thirsty():
    """Канал ppk28 у мухи работает как детектор воды и открывается только
    при обезвоживании: сытая водой муха на лужу не реагирует."""
    pops = _pool().pops
    rt = Config().runtime
    dry = SensoryEncoder(pops, rt)
    dry.thirst = 0.95
    dry.encode(Percept.from_json({"view_w": 15, "view_h": 15, "water_near": 1.0,
                                  "hydration": 0.05}))
    wet = SensoryEncoder(pops, rt)
    wet.encode(Percept.from_json({"view_w": 15, "view_h": 15, "water_near": 1.0,
                                  "hydration": 1.0}))
    assert dry.last_drive["gust_water"] > wet.last_drive["gust_water"] * 2.5


def test_thirsty_fly_finds_and_drinks_water():
    pool = _pool()
    a = Arena(seed=3, clutter=6)
    a.water = (a.fx, a.fy + 5)
    ag = pool.get("thirst")
    ag.encoder.thirst = 0.95
    for _ in range(30):
        ag.step(a.percept(-1))
    ag.encoder.thirst = 0.95
    before = ag.encoder.thirst
    for t in range(60):
        a.apply(ag.step(a.percept(t)))
    assert a.drinks >= 1, a.drinks
    assert ag.encoder.thirst < before - 0.15, (before, ag.encoder.thirst)


def test_sated_fly_ignores_the_puddle():
    pool = _pool()
    a = Arena(seed=3, clutter=6)
    a.water = (a.fx, a.fy + 5)
    ag = pool.get("sated")
    ag.encoder.thirst = 0.0
    for t in range(60):
        p = a.percept(t)
        p["hydration"] = 1.0           # игра говорит, что муха напоена
        a.apply(ag.step(p))
    assert a.drinks == 0, a.drinks


# ---------------------------------------------------------------------------
# Сомато­топия щетинок
# ---------------------------------------------------------------------------
def test_grooms_exactly_the_touched_side():
    for bearing, want in ((0.0, "голова"), (90.0, "правый бок"),
                          (180.0, "брюшко"), (270.0, "левый бок")):
        dec = MotorDecoder(spontaneous=False, fallback=False)
        _quiet(dec)
        a = dec.decode({k: 5.0 for k in KEYS}, None,
                       contact=0.8, touch_bearing=bearing)
        assert a.act == "groom" and a.part == want, (bearing, a.act, a.part)


def test_grooming_hierarchy_is_anterior_first():
    """У дрозофилы груминг идёт спереди назад: раздражение головы подавляет
    чистку брюшка, но не наоборот."""
    dec = MotorDecoder(spontaneous=False, fallback=False)
    _quiet(dec)
    a = dec.decode({k: 5.0 for k in KEYS}, None,
                   contact=0.9, touch_bearing=180.0, bitter=0.9)
    assert a.part == "голова", a.part


def test_grooming_outlasts_the_touch():
    """След от касания затухает не мгновенно: муха чистится ещё несколько
    секунд после того, как её отпустили."""
    dec = MotorDecoder(spontaneous=False, fallback=False)
    _quiet(dec)
    dec.decode({k: 5.0 for k in KEYS}, None, contact=0.9, touch_bearing=270.0)
    after = [dec.decode({k: 5.0 for k in KEYS}, None).part for _ in range(6)]
    assert after[0] == "левый бок", after
    assert None in after, "след должен закончиться"


def test_untouched_fly_does_not_groom_a_part():
    dec = MotorDecoder(spontaneous=False, fallback=False)
    _quiet(dec)
    parts = [dec.decode({k: 5.0 for k in KEYS}, None).part for _ in range(10)]
    assert all(p is None for p in parts), parts


def test_arena_body_touch_survives_turning():
    """Касание за бок ТЕЛА должно оставаться тем же боком при любом
    развороте — иначе сомато­топию нельзя проверить."""
    a = Arena(seed=1)
    for d in (1, 2, 4, 8):
        a.fdir = d
        a.touch_body(-1, 0)
        p = Percept.from_json(a.percept(0))
        assert p.touch_fx == -1 and p.touch_fy == 0, (d, p.touch_fx, p.touch_fy)
        assert 260 < (p.touch_bearing() or 0) < 280, (d, p.touch_bearing())


# ---------------------------------------------------------------------------
# Отрицательный геотаксис
# ---------------------------------------------------------------------------
def test_fly_climbs_the_furniture():
    pool = _pool()
    a = Arena(seed=3, clutter=6)
    a.climbable = (a.fx - 4, a.fy + 2)
    ag = pool.get("climb")
    for _ in range(30):
        ag.step(a.percept(-1))
    for t in range(90):
        a.apply(ag.step(a.percept(t)))
    assert a.climbs >= 1, a.climbs


def test_startle_raises_the_climbing_urge():
    """После испуга муха ползёт вверх — на этом построен стандартный тест
    на локомоцию (RING assay)."""
    dec = MotorDecoder(spontaneous=False, fallback=False)
    _quiet(dec)
    calm = dec.climb_urge
    for _ in range(4):
        dec.decode({**{k: 5.0 for k in KEYS}, "threat": 400.0}, None)
    assert dec.climb_urge > calm + 0.2, (calm, dec.climb_urge)


def test_being_high_up_satisfies_the_urge():
    dec = MotorDecoder(spontaneous=False, fallback=False)
    _quiet(dec)
    dec.climb_urge = 1.0
    for _ in range(6):
        dec.decode({k: 5.0 for k in KEYS}, None, on_high=1.0)
    assert dec.climb_urge < 0.2, dec.climb_urge


# ---------------------------------------------------------------------------
# Конкуренция мотиваций
# ---------------------------------------------------------------------------
def test_thirst_outranks_hunger():
    """Обезвоживание убивает быстрее голода, и у мухи жажда перебивает еду."""
    dec = MotorDecoder(spontaneous=False, fallback=False)
    _quiet(dec)
    g = dec._pick_goal(
        z={k: 0.0 for k in KEYS}, hunger=0.8, taste=0.0, food_near=0.7,
        food_bearing=0.0, thirst=0.8, water_near=0.7, water_bearing=90.0,
        booze_near=0.0, booze_bearing=None, drunk=0.0,
        climb_near=0.0, climb_bearing=None, on_high=0.0)
    assert g.name == "вода", g


def test_climbing_yields_to_everything():
    dec = MotorDecoder(spontaneous=False, fallback=False)
    _quiet(dec)
    dec.climb_urge = 1.0
    g = dec._pick_goal(
        z={k: 0.0 for k in KEYS}, hunger=0.7, taste=0.0, food_near=0.7,
        food_bearing=0.0, thirst=0.0, water_near=0.0, water_bearing=None,
        booze_near=0.0, booze_bearing=None, drunk=0.0,
        climb_near=0.9, climb_bearing=90.0, on_high=0.0)
    assert g.name == "еда", g
