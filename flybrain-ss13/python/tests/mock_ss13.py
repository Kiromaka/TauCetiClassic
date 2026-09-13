#!/usr/bin/env python3
"""Мок игрового сервера: крошечная станция на сетке.

Говорит с демоном ровно тем же протоколом, что и DM-модуль, включая
упаковку полей зрения в строки цифр. Нужен, чтобы отлаживать мозг мухи,
не поднимая BYOND.

    python3 tests/mock_ss13.py --ticks 200 --url http://127.0.0.1:5665
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request

NORTH, SOUTH, EAST, WEST = 1, 2, 4, 8
DVEC = {NORTH: (0, 1), SOUTH: (0, -1), EAST: (1, 0), WEST: (-1, 0)}
RADIUS = 7
SIDE = RADIUS * 2 + 1


def pack(vals) -> str:
    # та же семантика округления, что и у round() в DM (0.5 вверх)
    import math
    return "".join(chr(48 + max(0, min(9, int(math.floor(v * 9 + 0.5)))))
                   for v in vals)


class Station:
    """Комната с колоннами, лампой, куском еды и бродячим NPC."""

    def __init__(self, w=41, h=41, seed=0):
        self.rng = random.Random(seed)
        self.w, self.h = w, h
        self.wall = [[x in (0, w - 1) or y in (0, h - 1) for x in range(w)]
                     for y in range(h)]
        for _ in range(28):
            cx, cy = self.rng.randrange(2, w - 2), self.rng.randrange(2, h - 2)
            self.wall[cy][cx] = True
        self.lamp = (w // 2 + 6, h // 2 + 5)
        self.food = (w // 2 - 7, h // 2 + 3)
        self.npc = [w // 2 + 4, h // 2 - 6]
        self.fx, self.fy = w // 2, h // 2
        self.fdir = NORTH
        self.bumps = 0
        self.steps = 0
        self.visited = set()

    # ------------------------------------------------------------------
    def light_at(self, x, y):
        if not (0 <= x < self.w and 0 <= y < self.h):
            return 0.0
        d = abs(x - self.lamp[0]) + abs(y - self.lamp[1])
        return max(0.12, min(1.0, 1.3 - d / 14.0))

    def to_fly(self, dx, dy):
        if self.fdir == NORTH:   fx, fy = dx, dy
        elif self.fdir == SOUTH: fx, fy = -dx, -dy
        elif self.fdir == EAST:  fx, fy = -dy, dx
        else:                    fx, fy = dy, -dx
        col, row = fx + RADIUS, RADIUS - fy
        if not (0 <= col < SIDE and 0 <= row < SIDE):
            return -1
        return row * SIDE + col

    # ------------------------------------------------------------------
    def percept(self, tick, secret):
        light = [0.0] * (SIDE * SIDE)
        solid = [1.0] * (SIDE * SIDE)
        mobs = [0.0] * (SIDE * SIDE)
        items = [0.0] * (SIDE * SIDE)
        for dy in range(-RADIUS, RADIUS + 1):
            for dx in range(-RADIUS, RADIUS + 1):
                i = self.to_fly(dx, dy)
                if i < 0:
                    continue
                x, y = self.fx + dx, self.fy + dy
                if not (0 <= x < self.w and 0 <= y < self.h):
                    continue
                light[i] = self.light_at(x, y)
                solid[i] = 1.0 if self.wall[y][x] else 0.0
                if [x, y] == self.npc:
                    mobs[i] = 1.0
                if (x, y) == self.food:
                    items[i] = 1.0
        fd = abs(self.fx - self.food[0]) + abs(self.fy - self.food[1])
        return {
            "mob": "mock", "secret": secret, "tick": tick,
            "view_w": SIDE, "view_h": SIDE,
            "light": pack(light), "solid": pack(solid),
            "mobs": pack(mobs), "items": pack(items),
            # сытость медленно падает, изредка прилетает урон —
            # чтобы каналы голода и боли были не всегда нулевыми
            "health": 1.0 - min(0.4, tick / 2000.0),
            "pain": 0.7 if (tick % 137 == 0 and tick) else 0.0,
            "nutrition": max(60.0, 440.0 - tick * 0.6),
            "suffocate": 0.0,
            "temp": 293.0, "pressure": 101.3, "co2": 0.0, "toxin": 0.0,
            "food_near": max(0.0, 1 - fd / 8.0),
            "food_touch": 1.0 if fd == 0 else 0.0,
            "bitter": 0.0, "wind": 0.0, "sound": 0.0,
            "contact": 0.7 if self.bumped else 0.0,
            "on_fire": 0, "stunned": 0,
        }

    bumped = False

    # ------------------------------------------------------------------
    def apply(self, act):
        self.bumped = False
        d = int(act.get("dir") or self.fdir)
        if d in DVEC:
            self.fdir = d
        mv = int(act.get("move") or 0)
        if mv:
            vx, vy = DVEC[self.fdir]
            if mv < 0:
                vx, vy = -vx, -vy
            nx, ny = self.fx + vx, self.fy + vy
            if 0 <= nx < self.w and 0 <= ny < self.h and not self.wall[ny][nx]:
                self.fx, self.fy = nx, ny
                self.steps += 1
                self.visited.add((nx, ny))
            else:
                self.bumps += 1
                self.bumped = True
        # NPC слегка бродит
        if self.rng.random() < 0.3:
            ax, ay = self.rng.choice(list(DVEC.values()))
            nx, ny = self.npc[0] + ax, self.npc[1] + ay
            if 0 <= nx < self.w and 0 <= ny < self.h and not self.wall[ny][nx]:
                self.npc = [nx, ny]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5665")
    ap.add_argument("--secret", default="")
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--hz", type=float, default=0.0,
                    help="ограничить темп; 0 = гнать как получится")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    st = Station()
    lat = []
    states = {}
    for t in range(a.ticks):
        body = json.dumps(st.percept(t, a.secret)).encode()
        req = urllib.request.Request(a.url.rstrip("/") + "/tick", data=body,
                                     method="POST",
                                     headers={"Content-Type": "application/json",
                                              "X-Flybrain-Secret": a.secret})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                act = json.loads(r.read())
        except Exception as exc:  # noqa: BLE001
            print(f"тик {t}: не достучался до демона: {exc}", file=sys.stderr)
            return 1
        lat.append(time.perf_counter() - t0)
        st.apply(act)
        s = (act.get("dbg") or {}).get("state", "?")
        states[s] = states.get(s, 0) + 1
        if not a.quiet and t % 20 == 0:
            b = act.get("brain", {})
            print(f"t={t:4d} поз=({st.fx},{st.fy}) dir={act.get('dir')} "
                  f"move={act.get('move'):+d} состояние={s:<10} "
                  f"спайков={b.get('spikes','?'):>6} "
                  f"мозг={b.get('ms','?')}мс/{b.get('wall_ms','?')}мс")
        if a.hz:
            time.sleep(max(0.0, 1.0 / a.hz - (time.perf_counter() - t0)))

    lat.sort()
    print(f"\nтиков: {a.ticks}   шагов: {st.steps}   упёрлась в стену: {st.bumps} раз")
    print(f"уникальных клеток посещено: {len(st.visited)}")
    print(f"round-trip: медиана {lat[len(lat)//2]*1000:.0f} мс, "
          f"p95 {lat[int(len(lat)*0.95)]*1000:.0f} мс")
    print("распределение состояний: " +
          ", ".join(f"{k}={v}" for k, v in sorted(states.items(), key=lambda t: -t[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
