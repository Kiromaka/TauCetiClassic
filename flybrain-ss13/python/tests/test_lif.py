"""Проверки LIF-движка: соответствие уравнениям и предохранители."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flybrain.config import ModelParams, RuntimeParams          # noqa: E402
from flybrain.connectome import Connectome, make_synthetic       # noqa: E402
from flybrain.lif import LIFEngine                               # noqa: E402


def _empty(n=64):
    return Connectome(
        indptr=np.zeros(n + 1, dtype=np.int64),
        indices=np.empty(0, np.int32), weights=np.empty(0, np.float32),
        root_ids=np.arange(n, dtype=np.uint64),
        cell_type=np.array([""] * n, dtype="<U48"),
        hb_type=np.array([""] * n, dtype="<U48"),
        klass=np.array([""] * n, dtype="<U48"),
        super_class=np.array(["central"] * n, dtype="<U48"),
        side=np.array([""] * n, dtype="<U48"),
        nt=np.array(["ACH"] * n, dtype="<U16"),
        xyz=np.zeros((n, 3), np.float32), meta={})


def test_quiescent_without_input():
    """Без накачки и без фона сеть обязана молчать."""
    eng = LIFEngine(_empty(), ModelParams(),
                    RuntimeParams(dt=0.1, background_hz=0.0, homeostasis=False))
    r = eng.run(200.0)
    assert r.total_spikes == 0
    assert np.allclose(eng.v, ModelParams().v_rest, atol=1e-3)


def test_membrane_relaxes_to_rest_plus_g():
    """v должен стремиться к v_rest + g: это и есть dv/dt=(v_0-v+g)/tau_m."""
    p = ModelParams()
    eng = LIFEngine(_empty(4), p,
                    RuntimeParams(dt=0.05, background_hz=0.0, homeostasis=False))
    eng.g[:] = 3.0                      # ниже порога (порог - покой = 7 мВ)
    # синаптическое затухание отключаем, чтобы проверить именно мембрану
    eng.decay_g = np.float32(1.0)
    eng.run(200.0)
    assert abs(float(eng.v[0]) - (p.v_rest + 3.0)) < 0.05


def test_threshold_and_refractory():
    """Под сильной накачкой частота упирается в рефрактерный период.

    Проверять постоянным g нельзя: по правилу сброса Shiu et al. спайк
    обнуляет и v, и g, так что без входящих спайков нейрон замолкает.
    Поэтому гоним непрерывный пуассоновский вход.
    """
    p = ModelParams()
    rt = RuntimeParams(dt=0.1, background_hz=0.0, homeostasis=False,
                       sensory_max_hz=250.0, input_drive_mv=60.0)
    max_rate = 1000.0 / p.t_refractory          # жёсткий потолок ~454 Гц
    rates = []
    for drive in (20.0, 60.0, 200.0):
        rt = RuntimeParams(dt=0.1, background_hz=0.0, homeostasis=False,
                           sensory_max_hz=250.0, input_drive_mv=drive)
        eng = LIFEngine(_empty(1), p, rt)
        r = eng.run(1000.0, np.array([0]), np.array([250.0]))
        rates.append(r.total_spikes)
        assert r.total_spikes <= max_rate * 1.02, (drive, r.total_spikes)
    # частота растёт с накачкой и упирается в мембранную постоянную,
    # а не в рефрактерность: при tau_m = 20 мс на набор 7 мВ уходит время
    assert rates[0] < rates[1] < rates[2], rates
    assert rates[-1] > 150, rates


def test_spike_resets_synaptic_variable():
    """Сброс по статье обнуляет и g тоже."""
    p = ModelParams()
    eng = LIFEngine(_empty(1), p,
                    RuntimeParams(dt=0.1, background_hz=0.0, homeostasis=False))
    eng.g[:] = 40.0
    eng.decay_g = np.float32(1.0)
    r = eng.run(20.0)
    assert r.total_spikes == 1                  # один спайк и тишина
    assert float(eng.g[0]) == 0.0


def test_synaptic_delay_applied():
    """Спайк нейрона 0 должен дойти до нейрона 1 примерно через t_dly."""
    p = ModelParams()
    n = 2
    cx = _empty(n)
    cx.indptr = np.array([0, 1, 1], dtype=np.int64)
    cx.indices = np.array([1], dtype=np.int32)
    cx.weights = np.array([30.0], dtype=np.float32)
    rt = RuntimeParams(dt=0.1, background_hz=0.0, homeostasis=False)
    eng = LIFEngine(cx, p, rt)
    eng.g[0] = 40.0
    eng.decay_g = np.float32(1.0)
    # порог выше покоя на 7 мВ, при g=40 это ~3.6 мс; берём чуть меньше,
    # чтобы спайк уже случился, а задержка ещё не истекла
    eng.run(4.0)
    assert float(eng.g[0]) == 0.0, "нейрон 0 должен был спайкнуть"
    before = float(eng.g[1])
    eng.run(3.0)                        # ждём дольше t_dly = 1.8 мс
    after = float(eng.g[1])
    assert before <= 1e-6 < after, (before, after)


def test_homeostasis_tames_runaway():
    """Гомеостаз должен сбивать разгон к целевой частоте."""
    cx = make_synthetic(4000, 60, seed=7)
    sens = np.flatnonzero(cx.super_class == "optic")
    rt_off = RuntimeParams(dt=0.5, homeostasis=False, background_hz=40.0,
                           background_synapses=60)
    rt_on = RuntimeParams(dt=0.5, homeostasis=True, target_rate_hz=2.0,
                          background_hz=40.0, background_synapses=60)
    hz = np.full(len(sens), 250.0)
    off = LIFEngine(cx, ModelParams(), rt_off, seed=1)
    on = LIFEngine(cx, ModelParams(), rt_on, seed=1)
    for _ in range(40):
        r_off = off.run(40.0, sens, hz)
        r_on = on.run(40.0, sens, hz)
    assert r_on.mean_rate_hz < r_off.mean_rate_hz or r_off.mean_rate_hz < 3.0
    assert r_on.mean_rate_hz < 8.0


def test_population_rates_are_per_neuron():
    cx = make_synthetic(3000, 40, seed=2)
    dn = np.flatnonzero(cx.super_class == "descending")
    eng = LIFEngine(cx, ModelParams(), RuntimeParams(dt=0.5))
    r = eng.run(100.0, record={"dn": dn})
    expect = r.pop_spikes["dn"] / len(dn) / 0.1
    assert abs(r.pop_rates["dn"] - expect) < 1e-6


def test_structured_fixture_is_selective():
    """В слоистой заглушке яркость слева должна тянуть за левый поворот."""
    cx = make_synthetic(20000, 80, seed=3)
    vl = np.flatnonzero((cx.klass == "visual") & (cx.side == "left"))
    vr = np.flatnonzero((cx.klass == "visual") & (cx.side == "right"))
    rec = {"tl": np.flatnonzero((cx.cell_type == "DNa01") & (cx.side == "left")),
           "tr": np.flatnonzero((cx.cell_type == "DNa01") & (cx.side == "right"))}
    out = {}
    for name, (a, b) in {"left": (0.9, 0.2), "right": (0.2, 0.9)}.items():
        eng = LIFEngine(cx, ModelParams(), RuntimeParams(dt=1.0), seed=1)
        idx = np.concatenate([vl, vr])
        hz = np.concatenate([np.full(len(vl), 250 * a), np.full(len(vr), 250 * b)])
        for _ in range(20):
            r = eng.run(40.0, idx, hz, rec)
        out[name] = (r.pop_rates["tl"], r.pop_rates["tr"])
    assert out["left"][0] > out["left"][1] * 1.5, out
    assert out["right"][1] > out["right"][0] * 1.5, out
