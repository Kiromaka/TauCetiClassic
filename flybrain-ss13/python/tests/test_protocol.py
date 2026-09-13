"""Протокол DM <-> Python.

Здесь лежит построчная транскрипция двух процедур из _flybrain.dm
(flybrain_pack и flybrain_cell_index) на Python. Скомпилировать DM в этом
окружении нельзя, поэтому тесты проверяют, что питоновская сторона понимает
ровно тот формат, который описан в DM, и что обе реализации разворота поля
зрения согласованы между собой.

Если правите _flybrain.dm — правьте и транскрипцию, тесты её сторожат.
"""
import json
import os
import sys
import threading
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flybrain.encoders import Percept                                # noqa: E402

NORTH, SOUTH, EAST, WEST = 1, 2, 4, 8
RADIUS = 7
SIDE = RADIUS * 2 + 1


# --- транскрипция /proc/flybrain_pack --------------------------------------
def dm_pack(vals):
    # ВАЖНО: round() в DM округляет 0.5 вверх, а round() в Python — к чётному.
    # Здесь нужна именно семантика DM, иначе транскрипция врёт на пол-уровня.
    import math
    return "".join(chr(48 + max(0, min(9, int(math.floor(v * 9 + 0.5)))))
                   for v in vals)


# --- транскрипция /proc/flybrain_cell_index --------------------------------
def dm_cell_index(dx, dy, facing):
    if facing == NORTH:
        fx, fy = dx, dy
    elif facing == SOUTH:
        fx, fy = -dx, -dy
    elif facing == EAST:
        fx, fy = -dy, dx
    elif facing == WEST:
        fx, fy = dy, -dx
    else:
        fx, fy = dx, dy
    col = fx + RADIUS
    row = RADIUS - fy
    if not (0 <= col < SIDE and 0 <= row < SIDE):
        return -1
    return row * SIDE + col


def test_pack_matches_percept_decoder():
    rng = np.random.default_rng(0)
    vals = rng.random(SIDE * SIDE)
    packed = dm_pack(vals)
    p = Percept.from_json({"view_w": SIDE, "view_h": SIDE, "light": packed})
    # упаковка квантует до 10 уровней, поэтому сверяем с округлением
    assert np.abs(p.light - np.round(vals * 9) / 9).max() < 1e-6


def test_pack_is_one_char_per_cell():
    assert len(dm_pack([0.0] * 225)) == 225
    assert dm_pack([0.0, 0.5, 1.0, 2.0, -1.0]) == "05990"


def test_fly_frame_puts_forward_up_and_left_left():
    """Клетка прямо перед мухой — центр верхней строки, при любом dir."""
    for facing in (NORTH, SOUTH, EAST, WEST):
        ahead = {NORTH: (0, 1), SOUTH: (0, -1), EAST: (1, 0), WEST: (-1, 0)}[facing]
        i = dm_cell_index(ahead[0] * RADIUS, ahead[1] * RADIUS, facing)
        assert i == 0 * SIDE + RADIUS, (facing, i)

        left = {NORTH: (-1, 0), SOUTH: (1, 0), EAST: (0, 1), WEST: (0, -1)}[facing]
        i = dm_cell_index(left[0] * RADIUS, left[1] * RADIUS, facing)
        assert i == RADIUS * SIDE + 0, (facing, i)


def test_self_cell_is_centre():
    for facing in (NORTH, SOUTH, EAST, WEST):
        assert dm_cell_index(0, 0, facing) == RADIUS * SIDE + RADIUS


def test_out_of_field_rejected():
    assert dm_cell_index(RADIUS + 1, 0, NORTH) == -1
    assert dm_cell_index(0, -(RADIUS + 3), EAST) == -1


def test_mock_and_dm_agree_on_view_transform():
    """Мок SS13 и DM должны раскладывать поле зрения одинаково."""
    from mock_ss13 import Station
    st = Station()
    for facing in (NORTH, SOUTH, EAST, WEST):
        st.fdir = facing
        for dx in range(-RADIUS, RADIUS + 1):
            for dy in range(-RADIUS, RADIUS + 1):
                assert st.to_fly(dx, dy) == dm_cell_index(dx, dy, facing), \
                    (facing, dx, dy)


def test_server_roundtrip_in_process():
    """Полный круг: пакет -> сервер -> моторная команда."""
    from flybrain import server as srv
    from flybrain.config import Config
    from flybrain.connectome import make_synthetic

    cfg = Config.load()
    cfg.server.secret = "unit"
    cfg.server.port = 5699
    cfg.runtime.brain_ms_per_tick = 20.0
    cfg.runtime.dt = 1.0
    from flybrain.agent import BrainPool
    srv.CFG = cfg
    srv.POOL = BrainPool(cfg, cx=make_synthetic(4000, 40, seed=4))
    httpd = __import__("http.server", fromlist=["ThreadingHTTPServer"]) \
        .ThreadingHTTPServer(("127.0.0.1", cfg.server.port), srv.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        body = json.dumps({
            "mob": "unit-fly", "secret": "unit", "tick": 1,
            "view_w": SIDE, "view_h": SIDE,
            "light": dm_pack(np.full(225, 0.5)),
            "solid": dm_pack(np.zeros(225)),
            "mobs": dm_pack(np.zeros(225)),
            "items": dm_pack(np.zeros(225)),
            "hunger": 0.5, "temp": 293.0,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{cfg.server.port}/tick", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            act = json.loads(r.read())
        assert act["dir"] in (NORTH, SOUTH, EAST, WEST)
        assert act["move"] in (-1, 0, 1)
        assert act["run"] in (0, 1)
        assert "brain" in act and act["brain"]["ms"] > 0
        assert "dbg" in act and "z" in act["dbg"]

        # неверный секрет должен отлетать
        bad = json.dumps({"mob": "x", "secret": "nope"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{cfg.server.port}/tick", data=bad, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("сервер принял неверный секрет")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
    finally:
        httpd.shutdown()


def test_cloud_gives_real_coordinates_and_matching_indices():
    """3D-панель и растр должны показывать ОДНИ И ТЕ ЖЕ нейроны: подсветка
    в кадре — это индексы в массиве live, а не отдельный источник."""
    import numpy as np
    from flybrain.agent import BrainPool
    from flybrain.config import Config
    from flybrain.connectome import make_synthetic

    cfg = Config()
    cfg.runtime.brain_ms_per_tick = 20.0
    ag = BrainPool(cfg, make_synthetic(4000, 25, seed=21)).get("c")
    out = ag.step({"view_w": 15, "view_h": 15, "light": "5" * 225})
    cl = ag.cloud(shape_n=500)

    assert cl["real_coords"] is True
    assert len(cl["shape"]) == 500
    assert len(cl["live"]) == len(ag.raster_bands["idx"])
    assert len(cl["band"]) == len(cl["live"])
    assert set(cl["band"]) <= {0, 1, 2}
    # координаты нормированы и центрированы, иначе панель покажет точку
    pts = np.array(cl["shape"], dtype=float)
    assert pts.shape[1] == 3
    assert abs(float(pts.mean())) < 0.5, pts.mean()
    assert 0.2 < float(np.abs(pts).max()) < 12.0, np.abs(pts).max()
    # оси упорядочены по размаху: первая — самая широкая
    sd = pts.std(0)
    assert sd[0] >= sd[1] >= sd[2] - 1e-9, sd

    frame = ag.viz_frame(full=True)
    rows = frame["raster_rows"]
    assert rows == len(cl["live"])
    for col in frame["raster"]:
        assert all(0 <= i < rows for i in col)
    assert "thresholds" in frame and frame["thresholds"]["act"] > 0


def test_cloud_survives_a_connectome_without_coordinates():
    """Выгрузка без координат не должна ронять панель — только честно
    сообщать, что форма условная."""
    import numpy as np
    from flybrain.agent import BrainPool
    from flybrain.config import Config
    from flybrain.connectome import make_synthetic

    cx = make_synthetic(3000, 20, seed=22)
    cx.xyz = np.zeros_like(cx.xyz)
    cfg = Config()
    cfg.runtime.brain_ms_per_tick = 20.0
    ag = BrainPool(cfg, cx).get("c2")
    ag.step({"view_w": 15, "view_h": 15})
    cl = ag.cloud(shape_n=300)
    assert cl["real_coords"] is False
    assert len(cl["shape"]) == 300
    assert all(len(p) == 3 for p in cl["shape"])
