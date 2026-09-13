"""HTTP-сервер мозга. Только стандартная библиотека + numpy.

Горячий путь — POST /tick: в теле сенсорный пакет из DM, в ответе моторная
команда. Один round-trip на игровой тик, ничего дожидаться не нужно.

Прочее:
    GET  /status            состояние пула
    GET  /viz               дашборд
    GET  /viz/frame?id=..   кадр в JSON
    GET  /viz/stream?id=..  тот же кадр потоком (SSE)
    POST /reset             сброс мембран агента
    POST /drop              забыть агента
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse, parse_qs

from .agent import BrainPool, FlyAgent
from .config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
POOL: Optional[BrainPool] = None
CFG: Optional[Config] = None
STARTED = time.time()
COUNTERS = {"ticks": 0, "errors": 0, "wall_s": 0.0}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"      # keep-alive: экономит ~1 мс на тик
    server_version = "flybrain/1.0"

    # ------------------------------------------------------------------
    def log_message(self, fmt, *args):
        if os.environ.get("FLYBRAIN_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _auth(self, data: dict) -> bool:
        want = CFG.server.secret
        if not want or want == "":
            return True
        got = data.get("secret") or self.headers.get("X-Flybrain-Secret", "")
        return str(got) == str(want)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            # DM может прислать form-urlencoded — принимаем и такое
            from urllib.parse import parse_qsl
            d = dict(parse_qsl(raw.decode("utf-8", "replace")))
            if "json" in d:
                return json.loads(d["json"])
            return d

    # ------------------------------------------------------------------
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self._body()
            if not self._auth(data):
                return self._json(403, {"error": "bad secret"})

            if path == "/tick":
                mob = str(data.get("mob") or data.get("id") or "fly")
                t0 = time.perf_counter()
                out = POOL.get(mob).step(data)
                COUNTERS["ticks"] += 1
                COUNTERS["wall_s"] += time.perf_counter() - t0
                return self._json(200, out)

            if path == "/reset":
                POOL.get(str(data.get("mob", "fly"))).engine.reset()
                return self._json(200, {"ok": 1})

            if path == "/drop":
                POOL.drop(str(data.get("mob", "fly")))
                return self._json(200, {"ok": 1})

            return self._json(404, {"error": "no such endpoint"})
        except Exception as exc:  # noqa: BLE001
            COUNTERS["errors"] += 1
            traceback.print_exc()
            return self._json(500, {"error": str(exc)})

    # ------------------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/viz", "/viz/"):
                p = os.path.join(HERE, "viz", "dashboard.html")
                with open(p, "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")

            if u.path == "/status":
                up = time.time() - STARTED
                st = POOL.stats()
                st.update({
                    "uptime_s": round(up, 1),
                    "ticks": COUNTERS["ticks"],
                    "errors": COUNTERS["errors"],
                    "avg_tick_ms": round(
                        COUNTERS["wall_s"] / max(1, COUNTERS["ticks"]) * 1000, 2),
                    "backend": next(iter(POOL.agents.values())).engine.backend
                               if POOL.agents else CFG.runtime.backend,
                    "dt_ms": CFG.runtime.dt,
                    "weight_scale": CFG.runtime.weight_scale,
                    "brain_ms_per_tick": CFG.runtime.brain_ms_per_tick,
                    "config_file": CFG.loaded_from,
                    "projection": CFG.runtime.projection,
                    "diag": {k: a.diagnose() for k, a in
                             list(POOL.agents.items())[:8]},
                })
                return self._json(200, st)

            if u.path == "/viz/cloud":
                mob = (q.get("id") or ["fly"])[0]
                n = int((q.get("n") or ["9000"])[0])
                return self._json(200, POOL.get(mob).cloud(shape_n=n))

            if u.path == "/viz/hex":
                mob = (q.get("id") or ["fly"])[0]
                from .populations import HEX_SCALE
                return self._json(200, {"hex": POOL.get(mob).hex_xy(),
                                        "cell": round(HEX_SCALE, 6)})

            if u.path == "/viz/frame":
                mob = (q.get("id") or [None])[0]
                if mob is None:
                    mob = next(iter(POOL.agents), "fly")
                full = (q.get("full") or ["0"])[0] not in ("0", "", "false")
                return self._json(200, POOL.get(mob).viz_frame(full=full))

            if u.path == "/viz/stream":
                return self._sse(q)

            return self._json(404, {"error": "no such endpoint"})
        except FileNotFoundError:
            return self._json(404, {"error": "dashboard.html не найден"})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._json(500, {"error": str(exc)})

    # ------------------------------------------------------------------
    def _sse(self, q):
        mob = (q.get("id") or [None])[0]
        hz = float((q.get("hz") or ["8"])[0])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last = -1
        first = True
        try:
            while True:
                name = mob or next(iter(POOL.agents), None)
                if name is None:
                    time.sleep(0.5)
                    continue
                ag = POOL.get(name)
                if ag.ticks != last:
                    last = ag.ticks
                    payload = json.dumps(ag.viz_frame(full=first),
                                         ensure_ascii=False)
                    first = False
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                time.sleep(1.0 / max(1.0, hz))
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def _startup_check() -> None:
    """Проверить рабочую точку до того, как в игру зайдёт первый игрок.

    Мёртвая сеть и сеть в разгоне выглядят на дашборде одинаково — муха
    бегает, DN молчат, — и отличить их можно только измерением. Меряем сами
    и, если разрешено, сами же чиним и записываем в конфиг: ровно на шаге
    "перепишите значение руками" настройка и терялась.
    """
    from .autotune import autotune, probe_point
    r = CFG.runtime
    if not getattr(r, "startup_check", True):
        return
    t0 = time.time()

    def make(ws: float):
        import copy
        c = copy.copy(r)
        c.weight_scale = ws
        cfg2 = copy.copy(CFG)
        cfg2.runtime = c
        return FlyAgent("__probe__", cfg2, POOL.cx, POOL.pops, POOL.hex, seed=3)

    if getattr(r, "autotune", True):
        ws, probes = autotune(make, r.weight_scale, r.target_rate_hz,
                              ticks=getattr(r, "startup_ticks", 8))
    else:
        p = probe_point(lambda: make(r.weight_scale),
                        getattr(r, "startup_ticks", 8), r.weight_scale)
        ws, probes = r.weight_scale, [p]
    best = next((p for p in probes if abs(p.weight_scale - ws) < 1e-9), probes[0])

    if abs(ws - r.weight_scale) > 1e-9:
        r.weight_scale = ws
        POOL.cfg.runtime.weight_scale = ws
        POOL.agents.clear()          # мухи, созданные на старой точке, не годятся
        where = ""
        if getattr(r, "autotune_persist", True):
            try:
                where = CFG.patch_file({"runtime": {"weight_scale": float(ws)}})
            except OSError as e:
                where = f"(записать не удалось: {e})"
        print(f"[flybrain] взял weight_scale = {ws:g} "
              f"({best.mean_hz:.2f} Гц, живых DN-каналов {best.live_channels})"
              + (f", записал в {where}" if where else ""))
    else:
        print(f"[flybrain] самопроверка: {best.verdict}, "
              f"мозг {best.mean_hz:.2f} Гц, "
              f"тормоз {best.inhib_mv:.1f} мВ, живых DN-каналов "
              f"{best.live_channels}  ({time.time()-t0:.1f}s)")
        if not best.ok:
            print("[flybrain] рабочей точки не нашлось: муха поедет на "
                  "резервном контуре, а не на модели мозга.\n"
                  "           python3 scripts/explore.py --stats --paths")


def serve(cfg: Optional[Config] = None, cx=None) -> None:
    global POOL, CFG
    CFG = cfg or Config.load()
    # Печатаем действующие настройки: подобранный свипом weight_scale не
    # применяется, если конфиг не подхватился, и понять это по поведению
    # мухи невозможно.
    print(f"[flybrain] конфиг: {CFG.loaded_from or 'НЕ НАЙДЕН, всё по умолчанию'}")
    r = CFG.runtime
    print(f"[flybrain] weight_scale={r.weight_scale}  dt={r.dt} мс  "
          f"мозга на тик={r.brain_ms_per_tick} мс  "
          f"gain={CFG.model.gain}  гомеостаз={r.homeostasis} "
          f"(цель {r.target_rate_hz} Гц)")
    if r.weight_scale == 1.0 and not CFG.loaded_from:
        print("[flybrain] ВНИМАНИЕ: конфига нет и weight_scale = 1. Если вы "
              "подбирали его через calibrate.py --sweep,\n"
              "           значение не применилось — запустите свип ещё раз, "
              "он запишет файл сам.")
    print(f"[flybrain] сетчатка: проекция={r.projection}  поле глаза="
          f"{r.eye_fov_deg:g} гр. (перекрытие {r.eye_overlap_deg:g})  "
          f"ближняя граница={r.near_tiles:g} тайла")
    print(f"[flybrain] загружаю коннектом: {CFG.connectome_file}", flush=True)
    t0 = time.time()
    POOL = BrainPool(CFG, cx=cx)
    print(f"[flybrain] готово за {time.time()-t0:.1f}s")
    print(POOL.cx.describe())
    print(POOL.pops.summary())
    _startup_check()
    srv = ThreadingHTTPServer((CFG.server.host, CFG.server.port), Handler)
    srv.daemon_threads = True
    print(f"[flybrain] слушаю http://{CFG.server.host}:{CFG.server.port}  "
          f"(дашборд: /viz)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[flybrain] останавливаюсь")
        srv.shutdown()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Сервер мозга дрозофилы для SS13")
    ap.add_argument("--config", help="путь к json-конфигу")
    ap.add_argument("--connectome", help="путь к connectome.npz")
    ap.add_argument("--host"), ap.add_argument("--port", type=int)
    ap.add_argument("--secret")
    ap.add_argument("--dt", type=float, help="шаг интегрирования, мс")
    ap.add_argument("--brain-ms", type=float, help="мс мозгового времени на тик")
    ap.add_argument("--preset", choices=sorted(__import__(
        "flybrain.config", fromlist=["PRESETS"]).PRESETS),
        help="fast / balanced / realtime / faithful")
    ap.add_argument("--weight-scale", type=float, dest="weight_scale")
    ap.add_argument("--backend", choices=["auto", "numpy", "torch"])
    ap.add_argument("--device", choices=["cpu", "cuda"])
    ap.add_argument("--synthetic", type=int, nargs="?", const=139000,
                    help="не грузить коннектом, взять синтетику на N нейронов")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    if a.preset:
        from .config import PRESETS
        for k, v in PRESETS[a.preset].items():
            setattr(cfg.runtime, k, v)
        print(f"[flybrain] режим {a.preset}: {PRESETS[a.preset]}")
    if a.weight_scale is not None: cfg.runtime.weight_scale = a.weight_scale
    if a.connectome: cfg.connectome_file = a.connectome
    if a.host: cfg.server.host = a.host
    if a.port: cfg.server.port = a.port
    if a.secret: cfg.server.secret = a.secret
    if a.dt: cfg.runtime.dt = a.dt
    if a.brain_ms: cfg.runtime.brain_ms_per_tick = a.brain_ms
    if a.backend: cfg.runtime.backend = a.backend
    if a.device: cfg.runtime.device = a.device

    cx = None
    if a.synthetic:
        from .connectome import make_synthetic
        print(f"[flybrain] синтетический коннектом на {a.synthetic} нейронов "
              f"(настоящих данных FlyWire нет)")
        cx = make_synthetic(a.synthetic, 108, params=cfg.model)
    serve(cfg, cx=cx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
