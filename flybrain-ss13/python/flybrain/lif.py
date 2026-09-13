"""Событийно-управляемый LIF-движок для коннектома целого мозга.

Уравнения — как в Shiu et al., Nature 2024:

    dv/dt = (v_rest - v + g) / tau_m      (кроме рефрактерного периода)
    dg/dt = -g / tau_s
    спайк при v > v_th  ->  v = v_reset, g = 0, рефрактерность t_rfc
    приход спайка       ->  g += w  с задержкой t_dly

Интегрирование — экспоненциальный Эйлер по фиксированному шагу dt.
Дорогая часть — разноска спайков; она делается только по фактически
спайкнувшим нейронам (событийно), а не умножением на всю матрицу.

Стоимость шага:  O(n)  на плотные векторы  +  O(спайки * исходящая степень).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from .config import ModelParams, RuntimeParams
from .connectome import Connectome


@dataclass
class TickResult:
    """Итог одного шага симуляции длиной brain_ms."""
    brain_ms: float
    wall_s: float
    n_steps: int
    total_spikes: int
    mean_rate_hz: float
    pop_rates: Dict[str, float] = field(default_factory=dict)   # Гц на нейрон
    pop_spikes: Dict[str, int] = field(default_factory=dict)
    raster: Optional[np.ndarray] = None    # [n_sample] число спайков за тик
    clipped: bool = False                  # срабатывала ли защита от разгона
    inhib_mv: float = 0.0                  # текущий глобальный тормозной ток


class LIFEngine:
    def __init__(self,
                 cx: Connectome,
                 model: Optional[ModelParams] = None,
                 runtime: Optional[RuntimeParams] = None,
                 raster_sample: int = 1024,
                 seed: int = 0):
        self.cx = cx
        self.p = model or ModelParams()
        self.rt = runtime or RuntimeParams()
        self.n = cx.n
        self.rng = np.random.default_rng(seed)

        dt = float(self.rt.dt)
        self.dt = dt
        self.dtype = np.float32 if self.rt.dtype == "float32" else np.float64

        # предвычисленные коэффициенты
        self.decay_g = self.dtype(math.exp(-dt / self.p.tau_synapse))
        self.k_v = self.dtype(dt / self.p.tau_membrane)
        self.n_refrac = max(1, int(round(self.p.t_refractory / dt)))
        self.n_delay = max(1, int(round(self.p.t_delay / dt)))

        # состояние
        self.v = np.full(self.n, self.p.v_rest, dtype=self.dtype)
        self.g = np.zeros(self.n, dtype=self.dtype)
        self.refrac = np.zeros(self.n, dtype=np.int16)   # для torch-ветки
        self._tmp = np.empty(self.n, dtype=self.dtype)   # буфер мембраны
        self._spk = np.zeros(self.n, dtype=bool)         # маска спайков шага
        # Учёт популяций делаем по спайкнувшим, а не по популяциям: спайкнувших
        # за шаг десятки, а в зрительных популяциях бывает по четыре тысячи
        # нейронов, и читать их массивы каждый шаг — заметная доля тика.
        self._rec_masks: Dict[str, np.ndarray] = {}
        self._rec_key = None
        self._raster_pos = np.full(self.n, -1, dtype=np.int32)
        # Рефрактерность держим списком индексов, а не счётчиком на каждый
        # нейрон: при частотах в единицы герц одновременно «отдыхает» пара
        # сотен нейронов из 139 тысяч, и перебирать весь массив каждый шаг
        # незачем. Кольцо: в слот t кладём тех, кто спайкнул на шаге t,
        # через n_refrac+1 шагов слот переписывается — это и есть выход
        # из рефрактерности.
        self._ref_ring = [np.empty(0, dtype=np.int64)
                          for _ in range(self.n_refrac + 1)]
        self._ref_slot = 0
        # Раз в столько шагов зануляем денормализованные значения g.
        # Затухающая экспонента уводит их ниже 1e-38, а умножение на
        # денормалах на x86 идёт по микрокоду и медленнее раз в шестьдесят.
        self.flush_every = 128
        # Кольцевой буфер отложенной доставки. Держим не плотные массивы на
        # весь мозг, а пары (кому, сколько): в каждый слот пишет ровно один
        # шаг, и разложить пару тысяч значений через np.add.at на порядок
        # дешевле, чем гонять bincount по 139 тысячам элементов.
        self._pending = [None] * (self.n_delay + 1)
        self.acc = np.zeros((self.n_delay + 1, self.n), dtype=self.dtype) \
            if self.rt.backend == "torch" else None
        self.slot = 0

        # Вес виртуального сенсорного афферента считаем из желаемого
        # стационарного уровня g: g_ss = w * f * tau_s  ->
        #   w = input_drive_mv / (sensory_max_hz * tau_s / 1000)
        self.w_input = self.dtype(
            self.rt.input_drive_mv
            / max(1e-6, self.rt.sensory_max_hz * self.p.tau_synapse / 1000.0))
        self.w_background = self.dtype(
            self.rt.background_synapses * self.p.w_synapse * self.p.gain)

        # CSR
        self.indptr = cx.indptr.astype(np.int64)
        self.indices = cx.indices.astype(np.int64)     # int64 для bincount
        ws = float(getattr(self.rt, "weight_scale", 1.0))
        self.weights = (cx.weights.astype(self.dtype) if ws == 1.0
                        else (cx.weights * ws).astype(self.dtype))

        # выборка для растра
        if raster_sample and raster_sample < self.n:
            self.raster_idx = np.sort(self.rng.choice(self.n, raster_sample, replace=False))
        else:
            self.raster_idx = np.arange(self.n)
        self._sync_raster()

        # защита от разгона (не биология, инженерный предохранитель)
        self.rate_cap_hz = 200.0
        # глобальное тормозное управление усилением
        self.inhib = self.dtype(0.0)

        self.backend = self._pick_backend()
        self._t_brain = 0.0
        self._ema_wall_per_ms = 0.0

    # ------------------------------------------------------------------
    def _sync_raster(self) -> None:
        """Обратная карта нейрон -> строка растра, чтобы не читать выборку
        целиком на каждом шаге."""
        self._raster_pos = np.full(self.n, -1, dtype=np.int32)
        self._raster_pos[self.raster_idx] = np.arange(len(self.raster_idx),
                                                      dtype=np.int32)

    def _rec_membership(self, record) -> Dict[str, np.ndarray]:
        """Булевы маски принадлежности популяциям, с кэшем."""
        if not record:
            return {}
        key = tuple((k, len(v), int(v[0]) if len(v) else -1,
                     int(v[-1]) if len(v) else -1) for k, v in sorted(record.items()))
        if key != self._rec_key:
            self._rec_masks = {}
            for name, idxs in record.items():
                m = np.zeros(self.n, dtype=bool)
                if len(idxs):
                    m[np.asarray(idxs, dtype=np.int64)] = True
                self._rec_masks[name] = m
            self._rec_key = key
        return self._rec_masks

    # ------------------------------------------------------------------
    def _pick_backend(self) -> str:
        want = self.rt.backend
        if want in ("numpy",):
            return "numpy"
        try:
            import torch  # noqa: F401
        except Exception:
            if want == "torch":
                print("[flybrain] torch не найден, откатываюсь на numpy")
            return "numpy"
        if want in ("auto", "torch"):
            return self._init_torch()
        return "numpy"

    def _init_torch(self) -> str:
        import torch
        dev = self.rt.device
        if dev == "cuda" and not torch.cuda.is_available():
            print("[flybrain] cuda недоступна, torch на cpu")
            dev = "cpu"
        if dev == "cpu" and self.rt.backend == "auto":
            # на CPU numpy обычно не медленнее, а зависимостей меньше
            return "numpy"
        self.torch = torch
        self.tdev = torch.device(dev)
        td = torch.float32 if self.dtype is np.float32 else torch.float64
        self.ttype = td
        self.t_v = torch.from_numpy(self.v).to(self.tdev)
        self.t_g = torch.from_numpy(self.g).to(self.tdev)
        self.t_refrac = torch.from_numpy(self.refrac.astype(np.int32)).to(self.tdev)
        if self.acc is None:
            self.acc = np.zeros((self.n_delay + 1, self.n), dtype=self.dtype)
        self.t_acc = torch.zeros((self.n_delay + 1, self.n), dtype=td, device=self.tdev)
        self.t_indptr = torch.from_numpy(self.indptr).to(self.tdev)
        self.t_indices = torch.from_numpy(self.indices).to(self.tdev)
        self.t_weights = torch.from_numpy(self.weights.astype(
            np.float32 if td is torch.float32 else np.float64)).to(self.tdev)
        return "torch"

    # ------------------------------------------------------------------
    @property
    def brain_time_ms(self) -> float:
        return self._t_brain

    def reset(self) -> None:
        self.v[:] = self.p.v_rest
        self.g[:] = 0
        self.inhib = self.dtype(0.0)
        self.refrac[:] = 0
        self.acc[:] = 0
        self.slot = 0
        self._ref_ring = [np.empty(0, dtype=np.int64)
                          for _ in range(self.n_refrac + 1)]
        self._ref_slot = 0
        self._t_brain = 0.0
        if self.backend == "torch":
            self.t_v.fill_(self.p.v_rest)
            self.t_g.zero_()
            self.t_refrac.zero_()
            self.t_acc.zero_()

    # ------------------------------------------------------------------
    def run(self,
            brain_ms: float,
            input_idx: Optional[np.ndarray] = None,
            input_hz: Optional[np.ndarray] = None,
            record: Optional[Dict[str, np.ndarray]] = None,
            background_hz: Optional[float] = None) -> TickResult:
        """Прогнать brain_ms мозгового времени.

        input_idx/input_hz — разреженная накачка: индексы нейронов и частоты
        пуассоновского входа в Гц. record — словарь популяция -> индексы,
        по ним считаются спайки.
        """
        n_steps = max(1, int(round(brain_ms / self.dt)))
        t0 = time.perf_counter()

        bg = self.rt.background_hz if background_hz is None else background_hz
        if input_idx is None or len(input_idx) == 0:
            act_idx = np.empty(0, dtype=np.int64)
            act_hz = np.empty(0, dtype=np.float64)
        else:
            act_idx = np.asarray(input_idx, dtype=np.int64)
            act_hz = np.asarray(input_hz, dtype=np.float64)
            keep = act_hz > 0
            act_idx, act_hz = act_idx[keep], act_hz[keep]

        # матрица пуассоновских входов на весь тик — одним вызовом rng
        if act_idx.size:
            lam = np.clip(act_hz * self.dt / 1000.0, 0, 50.0)
            pois = self.rng.poisson(lam[None, :], size=(n_steps, act_idx.size))
        else:
            pois = None

        # фоновая накачка — редкая, поэтому считаем как общее число событий
        bg_lam = bg * self.dt / 1000.0 * self.n
        bg_total = self.rng.poisson(bg_lam * n_steps) if bg > 0 else 0

        if self.backend == "torch":
            res = self._run_torch(n_steps, act_idx, pois, bg_total, record)
        else:
            res = self._run_numpy(n_steps, act_idx, pois, bg_total, record)

        # подстройка тормозного тока по ошибке средней частоты
        if self.rt.homeostasis:
            err = res.mean_rate_hz - self.rt.target_rate_hz
            k = (self.rt.homeostasis_gain if err > 0
                 else getattr(self.rt, "homeostasis_release", 1.5))
            self.inhib = self.dtype(min(self.rt.homeostasis_max,
                                        max(0.0, float(self.inhib) + k * err)))

        self._t_brain += n_steps * self.dt
        res.brain_ms = n_steps * self.dt
        res.wall_s = time.perf_counter() - t0
        res.n_steps = n_steps
        if res.brain_ms > 0:
            w = 0.2
            self._ema_wall_per_ms = ((1 - w) * self._ema_wall_per_ms
                                     + w * res.wall_s / res.brain_ms)
        return res

    # ------------------------------------------------------------------
    def _run_numpy(self, n_steps, act_idx, pois, bg_total, record) -> TickResult:
        v, g, tmp, pending = self.v, self.g, self._tmp, self._pending
        indptr, indices, weights = self.indptr, self.indices, self.weights
        n, nslots = self.n, self.n_delay + 1
        decay_g, k_v, w_in = self.decay_g, self.k_v, self.w_input
        v_reset = self.dtype(self.p.v_reset)
        v_th = self.dtype(self.p.v_threshold)
        drive = self.dtype(self.p.v_rest - float(self.inhib))
        ring, R = self._ref_ring, self.n_refrac + 1
        empty = np.empty(0, dtype=np.int64)
        spk_mask = self._spk

        total_spikes = 0
        clipped = False
        raster = np.zeros(len(self.raster_idx), dtype=np.int32)
        rec_counts = {k: 0 for k in (record or {})}
        rec_masks = self._rec_membership(record)
        if len(self._raster_pos) != n or (
                len(self.raster_idx) and self._raster_pos[self.raster_idx[0]] < 0):
            self._sync_raster()
        raster_pos = self._raster_pos
        max_spikes_per_step = int(self.rate_cap_hz * n * self.dt / 1000.0) + 8

        if bg_total:
            bg_steps = self.rng.integers(0, n_steps, size=bg_total)
            bg_cells = self.rng.integers(0, n, size=bg_total)
            bg_order = np.argsort(bg_steps, kind="stable")
            bg_steps, bg_cells = bg_steps[bg_order], bg_cells[bg_order]
            bg_ptr = np.searchsorted(bg_steps, np.arange(n_steps + 1))
        else:
            bg_cells = None

        for s in range(n_steps):
            slot = self.slot
            # 1) затухание синаптической переменной и доставка отложенных
            np.multiply(g, decay_g, out=g)
            due = pending[slot]
            if due is not None:
                np.add.at(g, due[0], due[1])
                pending[slot] = None

            # раз в flush_every шагов выбрасываем денормалы
            if (s & (self.flush_every - 1)) == 0:
                np.multiply(g, np.abs(g) > self.dtype(1e-12), out=g)

            # 2) сенсорная накачка
            if pois is not None:
                row = pois[s]
                nzr = row.nonzero()[0]
                if nzr.size:
                    g[act_idx[nzr]] += row[nzr].astype(self.dtype) * w_in
            if bg_cells is not None:
                a, b = bg_ptr[s], bg_ptr[s + 1]
                if b > a:
                    g[bg_cells[a:b]] += self.w_background

            # 3) мембрана: v += (v_rest - v + g - inhib) * dt/tau_m,
            #    посчитано слитно в предвыделенный буфер, без временных массивов
            np.subtract(g, v, out=tmp)
            tmp += drive
            tmp *= k_v
            v += tmp

            # 4) рефрактерных держим на потенциале сброса — тогда они
            #    физически не могут перейти порог, и отдельная маска не нужна
            rslot = self._ref_slot
            for j in range(R):
                if j != rslot and ring[j].size:
                    v[ring[j]] = v_reset

            # 5) спайки
            src = np.flatnonzero(v > v_th)
            k = src.size
            if k > max_spikes_per_step:
                src = src[np.argsort(v[src])[-max_spikes_per_step:]]
                k = src.size
                clipped = True
            ring[rslot] = src if k else empty
            self._ref_slot = (rslot + 1) % R

            if k:
                v[src] = v_reset
                g[src] = 0
                total_spikes += k
                # всё считаем по спайкнувшим: их десятки, а популяции —
                # тысячи нейронов
                pos = raster_pos[src]
                hit = pos >= 0
                if hit.any():
                    np.add.at(raster, pos[hit], 1)
                for name, m in rec_masks.items():
                    rec_counts[name] += int(m[src].sum())

                # 6) разноска по CSR (событийно)
                starts = indptr[src]
                lens = indptr[src + 1] - starts
                tot = int(lens.sum())
                if tot:
                    cum = np.cumsum(lens)
                    base = starts.copy()
                    base[1:] -= cum[:-1]
                    flat = np.repeat(base, lens) + np.arange(tot)
                    dslot = (slot + self.n_delay) % nslots
                    pending[dslot] = (indices[flat], weights[flat])

            self.slot = (slot + 1) % nslots

        # для torch-ветки и для reset() держим счётчик согласованным
        return self._finish(n_steps, total_spikes, raster, rec_counts, record, clipped)

    # ------------------------------------------------------------------
    def _run_torch(self, n_steps, act_idx, pois, bg_total, record) -> TickResult:
        torch = self.torch
        dev, td = self.tdev, self.ttype
        v, g, refrac, acc = self.t_v, self.t_g, self.t_refrac, self.t_acc
        indptr, indices, weights = self.t_indptr, self.t_indices, self.t_weights
        n, nslots = self.n, self.n_delay + 1
        v_rest, v_reset, v_th = self.p.v_rest, self.p.v_reset, self.p.v_threshold

        t_act = torch.from_numpy(act_idx).to(dev) if act_idx.size else None
        t_pois = torch.from_numpy(pois.astype(np.float32)).to(dev, td) if pois is not None else None
        t_rast_idx = torch.from_numpy(self.raster_idx).to(dev)
        raster = torch.zeros(len(self.raster_idx), dtype=torch.int32, device=dev)
        t_rec = {k: torch.from_numpy(np.asarray(vv, dtype=np.int64)).to(dev)
                 for k, vv in (record or {}).items()}
        rec_counts = {k: 0 for k in t_rec}
        total_spikes = 0

        if bg_total:
            bg_steps = self.rng.integers(0, n_steps, size=bg_total)
            bg_cells = self.rng.integers(0, n, size=bg_total)
            o = np.argsort(bg_steps, kind="stable")
            bg_steps, bg_cells = bg_steps[o], bg_cells[o]
            bg_ptr = np.searchsorted(bg_steps, np.arange(n_steps + 1))
            t_bg = torch.from_numpy(bg_cells).to(dev)
        else:
            t_bg = None

        for s in range(n_steps):
            slot = self.slot
            g.mul_(self.decay_g).add_(acc[slot])
            acc[slot].zero_()
            if t_pois is not None:
                g.index_add_(0, t_act, t_pois[s] * float(self.w_input))
            if t_bg is not None:
                a, b = int(bg_ptr[s]), int(bg_ptr[s + 1])
                if b > a:
                    g.index_add_(0, t_bg[a:b],
                                 torch.full((b - a,), float(self.w_background),
                                            dtype=td, device=dev))
            active = refrac <= 0
            v.add_(torch.where(active, (v_rest - v + g - float(self.inhib)) * float(self.k_v),
                               torch.zeros((), dtype=td, device=dev)))
            v.masked_fill_(~active, v_reset)
            refrac.sub_(refrac.clamp(max=1))

            spk = (v > v_th) & active
            k = int(spk.sum().item())
            if k:
                src = spk.nonzero(as_tuple=True)[0]
                v[src] = v_reset
                g[src] = 0
                refrac[src] = self.n_refrac
                total_spikes += k
                raster.add_(spk[t_rast_idx].to(torch.int32))
                for name, idxs in t_rec.items():
                    rec_counts[name] += int(spk[idxs].sum().item())

                starts = indptr[src]
                lens = indptr[src + 1] - starts
                tot = int(lens.sum().item())
                if tot:
                    cum = torch.cumsum(lens, 0)
                    base = starts.clone()
                    base[1:] -= cum[:-1]
                    flat = (torch.repeat_interleave(base, lens)
                            + torch.arange(tot, device=dev))
                    dslot = (slot + self.n_delay) % nslots
                    acc[dslot].index_add_(0, indices[flat], weights[flat])
            self.slot = (slot + 1) % nslots

        self.v = v.detach().cpu().numpy()
        self.g = g.detach().cpu().numpy()
        return self._finish(n_steps, total_spikes,
                            raster.cpu().numpy(), rec_counts, record, False)

    # ------------------------------------------------------------------
    def _finish(self, n_steps, total_spikes, raster, rec_counts, record, clipped):
        sim_s = n_steps * self.dt / 1000.0
        pop_rates, pop_spikes = {}, {}
        for name, idxs in (record or {}).items():
            cnt = rec_counts[name]
            size = max(1, len(idxs))
            pop_spikes[name] = cnt
            pop_rates[name] = cnt / size / sim_s if sim_s > 0 else 0.0
        return TickResult(
            brain_ms=n_steps * self.dt, wall_s=0.0, n_steps=n_steps,
            total_spikes=total_spikes,
            mean_rate_hz=total_spikes / self.n / sim_s if sim_s > 0 else 0.0,
            pop_rates=pop_rates, pop_spikes=pop_spikes,
            raster=raster, clipped=clipped, inhib_mv=float(self.inhib))

    # ------------------------------------------------------------------
    def wall_per_brain_ms(self) -> float:
        """Сглаженная стоимость: секунд реального времени на 1 мс мозга."""
        return self._ema_wall_per_ms
