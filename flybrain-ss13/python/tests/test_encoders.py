"""Кодировщик сенсорики: упаковка, поля зрения, каналы."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flybrain.config import (RuntimeParams, default_motor_specs,     # noqa: E402
                             default_sensory_specs)
from flybrain.connectome import make_synthetic                        # noqa: E402
from flybrain.encoders import Percept, SensoryEncoder                 # noqa: E402
from flybrain.populations import resolve, hex_lattice                 # noqa: E402

CX = make_synthetic(6000, 40, seed=11)
POPS = resolve(CX, default_sensory_specs(), default_motor_specs())


def _enc():
    return SensoryEncoder(POPS, RuntimeParams())


def _blank(**kw):
    d = {"view_w": 15, "view_h": 15, "light": "0" * 225, "solid": "0" * 225,
         "mobs": "0" * 225, "items": "0" * 225}
    d.update(kw)
    return Percept.from_json(d)


def test_packed_string_roundtrip():
    packed = "".join(str(i % 10) for i in range(225))
    p = Percept.from_json({"view_w": 15, "view_h": 15, "light": packed})
    assert abs(p.light[0] - 0.0) < 1e-6
    assert abs(p.light[9] - 1.0) < 1e-6
    assert abs(p.light[5] - 5 / 9) < 1e-3


def test_list_form_also_accepted():
    p = Percept.from_json({"view_w": 15, "view_h": 15, "light": [0.5] * 225})
    assert p.light.shape == (225,) and abs(p.light[0] - 0.5) < 1e-6


def test_short_or_missing_arrays_are_padded():
    p = Percept.from_json({"view_w": 15, "view_h": 15, "light": "999"})
    assert p.light.shape == (225,)
    assert p.solid.shape == (225,)


def test_walls_darken_and_mobs_brighten():
    enc = _enc()
    dark = enc.image(_blank(light="9" * 225, solid="9" * 225))
    lit = enc.image(_blank(light="9" * 225))
    assert dark.mean() < lit.mean() * 0.3
    with_mob = enc.image(_blank(light="4" * 225, mobs="9" * 225))
    assert with_mob.mean() > enc.image(_blank(light="4" * 225)).mean()


def test_static_scene_adapts_away():
    """Сцена, замершая после изменения, должна терять контраст.

    Ниже тонического уровня (0.32) накачка не падает: у мухи ламинарные
    нейроны активны и на неподвижной картинке.
    """
    enc = _enc()
    flat = "3" * 225
    for _ in range(10):
        enc.encode(_blank(light=flat))
    tonic = enc.last_drive["vis_left"]
    bright = list(flat)
    for i in (30, 31, 45, 46, 60, 61):
        bright[i] = "9"
    changed = _blank(light="".join(bright))
    enc.encode(changed)
    peak = enc.last_drive["vis_left"]
    assert peak > tonic
    for _ in range(40):
        enc.encode(changed)
    assert enc.last_drive["vis_left"] < peak
    assert enc.last_drive["vis_left"] >= tonic - 1e-6


def test_moving_object_creates_drive():
    enc = _enc()
    base = ["3"] * 225
    for i in range(12):                      # прогреваем адаптацию
        enc.encode(_blank(light="".join(base)))
    quiet = enc.last_drive["vis_left"]
    moved = base.copy()
    for i in (7 * 15 + 1, 7 * 15 + 2, 8 * 15 + 1):   # столбцы 1-2: левое поле
        moved[i] = "9"
    enc.encode(_blank(light="".join(moved)))
    assert enc.last_drive["vis_left"] > quiet


def test_hunger_gates_food_smell():
    enc = _enc()
    enc.encode(_blank(food_near=1.0, hunger=0.0))
    low = enc.last_drive["olf_attractive"]
    enc.encode(_blank(food_near=1.0, hunger=1.0))
    assert enc.last_drive["olf_attractive"] > low


def test_channels_map_to_disjoint_neurons():
    enc = _enc()
    idx, hz = enc.encode(_blank(food_near=1, bitter=1, wind=1, temp=320))
    assert len(idx) == len(set(idx.tolist())), "нейроны не должны дублироваться"
    assert (hz >= 0).all() and hz.max() <= RuntimeParams().sensory_max_hz + 1e-6


def test_hex_projection_shape():
    enc = _enc()
    enc.encode(_blank(light="5" * 225))
    hexa = hex_lattice(15)
    assert enc.hex_projection(hexa).shape == (721,)


def test_eyes_split_the_field():
    """Разделение полей зависит от проекции.

    При grid глаз отсекается маской по столбцам сетки; при перспективе
    сектор задан азимутом, и маска только выбрасывала бы половину нейронов,
    поэтому она отключена, а разделение проверяется по средним столбцам.
    """
    from flybrain.config import RuntimeParams as RP
    g = SensoryEncoder(POPS, RP(projection="grid"))
    left = g.eye_cells["vis_left"].reshape(15, 15)
    right = g.eye_cells["vis_right"].reshape(15, 15)
    assert left[0, 0] and not left[0, 14]        # левый глаз видит левый край
    assert right[0, 14] and not right[0, 0]
    assert (left & right).any(), "спереди поля должны перекрываться"

    enc = _enc()                                  # перспектива
    assert enc.eye_cells["vis_left"].all(), "при перспективе маска не нужна"
    lc = enc.view_map["vis_left"] % 15
    rc = enc.view_map["vis_right"] % 15
    assert lc.mean() < 7.0 < rc.mean(), (lc.mean(), rc.mean())


def test_pain_leaves_a_trace():
    """Урон приходит дельтой за один тик — на поведение так не повлиять.

    Ноцицепция у мухи фазная, но с последействием: держим затухающий след
    примерно на секунду, иначе удар не успевает ничего изменить и его даже
    не видно на дашборде.
    """
    enc = _enc()
    enc.encode(_blank())
    assert enc.last_drive.get("nocic", 0) == 0.0
    enc.encode(_blank(pain=0.8))
    peak = enc.last_drive["nocic"]
    assert peak > 0.5
    trace = []
    for _ in range(5):
        enc.encode(_blank())
        trace.append(enc.last_drive["nocic"])
    assert trace[0] < peak, "след должен затухать"
    assert trace[0] > 0.2, "и не пропадать сразу же"
    assert trace == sorted(trace, reverse=True), trace


def test_contact_no_longer_masks_pain():
    """Раньше боль подмешивалась в bristle, и постоянные столкновения
    полностью её прятали."""
    enc = _enc()
    enc.encode(_blank(contact=0.7))
    quiet = enc.last_drive["nocic"]
    enc.encode(_blank(contact=0.7, pain=0.9))
    assert enc.last_drive["nocic"] > quiet + 0.4
    # касание при этом остаётся касанием и от боли не растёт
    a = _enc()
    a.encode(_blank(contact=0.7))
    b = _enc()
    b.encode(_blank(contact=0.7, pain=0.9))
    assert abs(a.last_drive["bristle"] - b.last_drive["bristle"]) < 1e-6


def test_hunger_is_its_own_channel():
    enc = _enc()
    enc.encode(_blank(hunger=0.0))
    assert enc.last_drive.get("hunger", None) == 0.0
    enc.encode(_blank(hunger=0.8))
    assert enc.last_drive["hunger"] > 0.7


def test_hunger_drives_appetite_without_food_around():
    """Даже когда еды рядом нет, голод должен поднимать аппетитивный путь —
    иначе внутреннее состояние вообще ни на что не влияет."""
    enc = _enc()
    enc.encode(_blank(hunger=0.0, food_near=0.0))
    fed = enc.last_drive["olf_attractive"]
    enc.encode(_blank(hunger=1.0, food_near=0.0))
    assert enc.last_drive["olf_attractive"] > fed + 0.2


def test_hunger_rescaled_from_raw_nutrition():
    """Свежий человек рождается с nutrition 330: он слегка голоден, а не на 40%."""
    from flybrain.encoders import NUTRITION_STARVING, NUTRITION_WELL_FED
    assert Percept.from_json({"nutrition": NUTRITION_WELL_FED}).hunger == 0.0
    assert Percept.from_json({"nutrition": NUTRITION_STARVING}).hunger == 1.0
    spawn = Percept.from_json({"nutrition": 330}).hunger
    assert 0.15 < spawn < 0.4, spawn
    # старый DM без поля nutrition должен продолжать работать
    assert abs(Percept.from_json({"hunger": 0.42}).hunger - 0.42) < 1e-6


def test_suffocation_reaches_co2_channel():
    enc = _enc()
    enc.encode(_blank())
    assert enc.last_drive["olf_co2"] == 0.0
    enc.encode(_blank(suffocate=1.0))
    assert enc.last_drive["olf_co2"] > 0.5


def test_empty_type_pattern_does_not_grab_whole_brain():
    """Откат по классу применим, только если класс задан. Иначе пустая
    спецификация выбирала ВЕСЬ мозг и канал накачивал 139 тысяч нейронов."""
    from flybrain.config import PopulationSpec
    from flybrain.populations import resolve
    spec = dict(default_sensory_specs())
    spec["ghost"] = PopulationSpec(match_type=r"ЗАВЕДОМО_НЕТ_ТАКОГО")
    pops = resolve(CX, spec, default_motor_specs())
    assert len(pops.sensory["ghost"]) == 0, len(pops.sensory["ghost"])


def _field_counts(projection):
    """Сколько омматидиев смотрит в каждую клетку поля, обоими глазами."""
    from flybrain.populations import build_view_map, eye_sector
    cnt = np.zeros(225)
    for eye in ("vis_left", "vis_right"):
        a0, a1 = eye_sector(eye)
        vm = build_view_map(POPS.sensory[eye], 15, 15, projection=projection,
                            azim_from_deg=a0, azim_to_deg=a1)
        cnt += np.bincount(vm, minlength=225)
    return cnt


_ROWS = np.arange(225) // 15
_COLS = np.arange(225) % 15
_DIST = np.hypot(_ROWS - 7, _COLS - 7)
_RINGS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 7.5))


def _by_distance(cnt):
    return [cnt[(_DIST >= lo) & (_DIST < hi)].mean() for lo, hi in _RINGS]


def test_perspective_gives_near_cells_more_ommatidia():
    """Главный зрительный признак для мухи — рост объекта при приближении.

    При раскладке тайл-в-тайл клетка в одном шаге и клетка в шести занимают
    одинаковое число фасеток, и надвигание не кодируется вообще.
    """
    grid = _by_distance(_field_counts("grid"))
    persp = _by_distance(_field_counts("perspective"))
    r_grid = grid[0] / max(grid[-1], 1e-9)
    r_persp = persp[0] / max(persp[-1], 1e-9)
    assert r_grid < 2.0, r_grid
    assert r_persp > 10.0, r_persp


def test_ommatidia_per_tile_falls_off_with_distance():
    """Число фасеток на тайл должно падать монотонно: иначе объект на
    подходе то растёт, то сжимается."""
    curve = _by_distance(_field_counts("perspective"))
    assert curve == sorted(curve, reverse=True), curve


def test_far_tiles_are_not_blind():
    """Честная проекция на пол (ground) при восьмистах омматидиях оставляет
    дальнее кольцо пустым: половина фасеток смотрит ближе одного тайла, и
    подходящий за пять клеток человек просто не виден. Логарифмическая
    раскладка расстояний это чинит, сохраняя надвигание."""
    ground = _field_counts("ground")
    persp = _field_counts("perspective")
    g_far = _by_distance(ground)[-1]
    p_far = _by_distance(persp)[-1]
    assert g_far < 1.0, g_far
    assert p_far > 2.0 * g_far, (g_far, p_far)
    assert int((persp > 0).sum()) > int((ground > 0).sum()), "поле должно быть шире"
    # и при этом надвигание не потеряно
    assert _by_distance(persp)[0] > 10 * p_far


def test_approaching_wall_grows_the_response():
    enc = _enc()

    def wall_at(dist):
        cells = ["5"] * 225
        row = 7 - dist
        if 0 <= row < 15:
            for c in range(5, 10):
                cells[row * 15 + c] = "0"
        return _blank(light="".join(cells))

    resp = []
    for d in (6, 4, 2, 1):
        enc._adapt = None
        for _ in range(3):
            enc.encode(wall_at(d))
        enc.encode(wall_at(max(0, d - 1)))
        resp.append(enc.last_drive["vis_left"] + enc.last_drive["vis_right"])
    assert resp == sorted(resp), f"ответ не растёт при приближении: {resp}"
    assert resp[-1] > resp[0] * 1.2, resp


def test_eyes_cover_overlapping_sectors():
    """Секторы глаз задаются азимутом, а не столбцами сетки: иначе половина
    нейронов глаза смотрит в чужую полусферу и выбрасывается."""
    from flybrain.populations import build_view_map
    left = build_view_map(POPS.sensory["vis_left"], 15, 15,
                          azim_from_deg=-85.0, azim_to_deg=20.0)
    right = build_view_map(POPS.sensory["vis_right"], 15, 15,
                           azim_from_deg=-20.0, azim_to_deg=85.0)
    lc, rc = left % 15, right % 15
    assert lc.mean() < 7.0, lc.mean()      # левый глаз смотрит влево
    assert rc.mean() > 7.0, rc.mean()
    assert set(lc.tolist()) & set(rc.tolist()), "перекрытия спереди нет"


def test_config_file_is_picked_up_without_explicit_flag():
    """Лежащий рядом flybrain.json раньше молча игнорировался, и подобранный
    свипом weight_scale не применялся."""
    import json as _json
    import os as _os
    import tempfile as _tf
    from flybrain.config import Config

    d = _tf.mkdtemp()
    with open(_os.path.join(d, "flybrain.json"), "w", encoding="utf-8") as fh:
        _json.dump({"runtime": {"weight_scale": 6.0}}, fh)
    old = _os.getcwd()
    try:
        _os.chdir(d)
        cfg = Config.load()
        assert cfg.runtime.weight_scale == 6.0, cfg.runtime.weight_scale
        assert cfg.loaded_from and cfg.loaded_from.endswith("flybrain.json")
    finally:
        _os.chdir(old)


def test_uniform_light_gives_no_direction():
    """Яркость входит контрастом по Веберу, а не абсолютным уровнем.
    Равномерно освещённая комната никуда тянуть не должна."""
    enc = _enc()
    for _ in range(20):
        enc.encode(_blank(light="9" * 225))
    bright = enc.last_drive["vis_left"]
    dark = _enc()
    for _ in range(20):
        dark.encode(_blank(light="0" * 225))
    assert abs(bright - dark.last_drive["vis_left"]) < 0.02, (bright, dark.last_drive)


def test_lamp_in_the_dark_pulls_to_its_side():
    """Лампа сбоку в темноте должна давать устойчивый перекос между
    глазами — и не уходить в ноль от адаптации, как раньше."""
    def lamp(col):
        c = ["0"] * 225
        for r in (6, 7, 8):
            c[r * 15 + col] = "9"
        return "".join(c)

    left = _enc()
    right = _enc()
    for _ in range(25):
        left.encode(_blank(light=lamp(5)))
        right.encode(_blank(light=lamp(9)))
    dl = left.last_drive["vis_left"] - left.last_drive["vis_right"]
    dr = right.last_drive["vis_left"] - right.last_drive["vis_right"]
    assert dl > 0.005, dl
    assert dr < -0.005, dr


def test_optics_blur_spreads_a_point_source():
    """Функция чувствительности омматидия шире шага решётки: точечный
    источник должен задевать соседние фасетки, иначе в пяти тайлах он
    попадает в одну из четырёхсот и тонет в тонике."""
    enc = _enc()
    spot = ["0"] * 225
    spot[6 * 15 + 6] = "9"
    enc.encode(_blank(light="".join(spot)))
    lit = int((enc._blur(enc.image(_blank(light="".join(spot)))) > 1e-3).sum())
    enc.blur = 0.0
    sharp = int((enc._blur(enc.image(_blank(light="".join(spot)))) > 1e-3).sum())
    assert sharp == 1, sharp
    assert lit >= 5, lit
