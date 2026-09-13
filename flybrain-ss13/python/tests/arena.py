"""Управляемая арена для поведенческих проб.

В отличие от mock_ss13 здесь можно поставить лампу, положить еду, ударить
муху с заданной стороны — и померить, что она сделает. Протокол тот же,
что у DM-модуля, включая упаковку поля зрения в строки цифр.
"""
from __future__ import annotations

import math
import random
from typing import Optional, Tuple

NORTH, SOUTH, EAST, WEST = 1, 2, 4, 8
DVEC = {NORTH: (0, 1), SOUTH: (0, -1), EAST: (1, 0), WEST: (-1, 0)}
RADIUS = 7
SIDE = RADIUS * 2 + 1


def pack(vals) -> str:
    return "".join(chr(48 + max(0, min(9, int(math.floor(v * 9 + 0.5)))))
                   for v in vals)


class Arena:
    def __init__(self, w=61, h=61, seed=0, ambient=0.10, walls=True,
                 clutter=0):
        self.rng = random.Random(seed)
        self.w, self.h = w, h
        self.ambient = ambient
        self.wall = [[walls and (x in (0, w - 1) or y in (0, h - 1))
                      for x in range(w)] for y in range(h)]
        # Колонны: без них все повторы одного сценария идут в одинаковой
        # обстановке, и разброс берётся только из шума самой сети.
        for _ in range(clutter):
            cx = self.rng.randrange(2, w - 2)
            cy = self.rng.randrange(2, h - 2)
            if abs(cx - w // 2) + abs(cy - h // 2) > 3:
                self.wall[cy][cx] = True
        self.fx, self.fy = w // 2, h // 2
        self.fdir = NORTH
        self.lamp: Optional[Tuple[int, int]] = None
        self.food: Optional[Tuple[int, int]] = None
        self.attacker: Optional[Tuple[int, int]] = None
        self.pending_pain = 0.0
        self.bitter = 0.0
        self.nutrition = 440.0
        # Новое: вода, выпивка, мебель. Хранятся так же, как еда — точкой на
        # карте, — чтобы сценарии оставались короткими.
        self.water: Optional[Tuple[int, int]] = None
        self.booze: Optional[Tuple[int, int]] = None
        self.climbable: Optional[Tuple[int, int]] = None
        self.drunk = 0.0
        self.on_high = 0.0
        self.drinks = 0            # сколько глотков сделала
        self.climbs = 0            # сколько раз залезла
        self.touch_dir: Optional[Tuple[int, int]] = None
        self.bumps = 0
        self.steps = 0
        self.path = [(self.fx, self.fy)]

    # ------------------------------------------------------------------
    def light_at(self, x, y):
        if self.lamp is None:
            return 0.55
        d = math.hypot(x - self.lamp[0], y - self.lamp[1])
        return max(self.ambient, min(1.0, 1.35 - d / 11.0))

    def to_fly(self, dx, dy):
        if self.fdir == NORTH:   fx, fy = dx, dy
        elif self.fdir == SOUTH: fx, fy = -dx, -dy
        elif self.fdir == EAST:  fx, fy = -dy, dx
        else:                    fx, fy = dy, -dx
        col, row = fx + RADIUS, RADIUS - fy
        if not (0 <= col < SIDE and 0 <= row < SIDE):
            return -1
        return row * SIDE + col

    def fly_frame(self, dx, dy):
        if self.fdir == NORTH:   return dx, dy
        if self.fdir == SOUTH:   return -dx, -dy
        if self.fdir == EAST:    return -dy, dx
        return dy, -dx

    # ------------------------------------------------------------------
    def percept(self, tick, secret=""):
        n = SIDE * SIDE
        light = [0.0] * n
        solid = [1.0] * n
        mobs = [0.0] * n
        items = [0.0] * n
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
                if self.food and (x, y) == self.food:
                    items[i] = 1.0
                if self.attacker and (x, y) == self.attacker:
                    mobs[i] = 1.0

        fd = (abs(self.fx - self.food[0]) + abs(self.fy - self.food[1])
              if self.food else 99)
        def near_and_frame(pt, radius=8.0):
            """Обратное расстояние и пеленг, как их считает DM."""
            if pt is None:
                return 0.0, 0, 0
            d = abs(self.fx - pt[0]) + abs(self.fy - pt[1])
            near = max(0.0, 1 - d / radius) if d <= 7 else 0.0
            fx, fy = self.fly_frame(pt[0] - self.fx, pt[1] - self.fy)
            return round(near, 2), fx, fy

        ffx = ffy = 0
        if self.food:
            ffx, ffy = self.fly_frame(self.food[0] - self.fx,
                                      self.food[1] - self.fy)
        w_near, w_fx, w_fy = near_and_frame(self.water)
        b_near, b_fx, b_fy = near_and_frame(self.booze)
        c_near, c_fx, c_fy = near_and_frame(self.climbable)
        t_fx = t_fy = 0
        if self.touch_dir:
            t_fx, t_fy = self.fly_frame(*self.touch_dir)
        tfx = tfy = 0
        tnear = 0.0
        if self.attacker:
            tfx, tfy = self.fly_frame(self.attacker[0] - self.fx,
                                      self.attacker[1] - self.fy)
            dist = math.hypot(tfx, tfy)
            tnear = max(0.0, min(1.0, 1 - dist / 8.0))

        pain = self.pending_pain
        self.pending_pain = 0.0
        return {
            "mob": "arena", "secret": secret, "tick": tick,
            "view_w": SIDE, "view_h": SIDE,
            "light": pack(light), "solid": pack(solid),
            "mobs": pack(mobs), "items": pack(items),
            "health": 1.0, "pain": pain, "nutrition": self.nutrition,
            "temp": 293.0, "pressure": 101.3, "co2": 0.0, "toxin": 0.0,
            "food_near": max(0.0, 1 - fd / 8.0) if self.food else 0.0,
            # В руках еда оказывается только после подбора; под ногами —
            # как только муха встала на клетку с едой (вкус лапками).
            "food_touch": 1.0 if self.food_held else 0.0,
            "food_underfoot": 1.0 if fd == 0 else 0.0,
            "food_fx": ffx, "food_fy": ffy,
            "water_near": w_near, "water_fx": w_fx, "water_fy": w_fy,
            "booze_near": b_near, "booze_fx": b_fx, "booze_fy": b_fy,
            "drunk": round(self.drunk, 2),
            "climb_near": c_near, "climb_fx": c_fx, "climb_fy": c_fy,
            "on_high": self.on_high,
            "touch_fx": t_fx, "touch_fy": t_fy,
            "bitter": self.bitter, "wind": 0.0, "sound": 0.0,
            "contact": (1.0 if getattr(self, "_touched", False)
                        else (0.7 if self._bumped else 0.0)),
            "on_fire": 0, "stunned": 0, "suffocate": 0.0,
            "threat_fx": tfx, "threat_fy": tfy, "threat_near": round(tnear, 2),
        }

    _bumped = False
    food_held = False
    eaten = 0

    # ------------------------------------------------------------------
    def apply(self, act):
        self._bumped = False
        self._touched = False
        # Питьё как в DM: вода снимает жажду, спирт копится в крови.
        a = act.get("act")
        if a == "drink":
            self.drinks += 1
            wd = (abs(self.fx - self.water[0]) + abs(self.fy - self.water[1])
                  if self.water else 99)
            bd = (abs(self.fx - self.booze[0]) + abs(self.fy - self.booze[1])
                  if self.booze else 99)
            if bd <= 1:
                self.drunk = min(1.0, self.drunk + 0.18)
            elif wd <= 1:
                pass          # жажду снимает питоновская сторона
        if a == "climb":
            cd = (abs(self.fx - self.climbable[0])
                  + abs(self.fy - self.climbable[1]) if self.climbable else 99)
            if cd <= 1:
                self.climbs += 1
                self.on_high = 1.0
        # Пищевой акт как в DM: сначала подбираем, потом едим.
        if act.get("act") == "eat":
            fd = (abs(self.fx - self.food[0]) + abs(self.fy - self.food[1])
                  if self.food else 99)
            if self.food_held:
                self.eaten += 1
                self.nutrition = min(550.0, self.nutrition + 60.0)
                if self.eaten >= 3:
                    self.food_held = False
                    self.food = None
            elif fd <= 1:
                self.food_held = True
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
            else:
                self.bumps += 1
                self._bumped = True
        self.path.append((self.fx, self.fy))

    # ------------------------------------------------------------------
    def hit(self, damage=0.8):
        """Ударить муху. Бьёт тот, кто стоит в self.attacker."""
        self.pending_pain = damage
        if self.attacker:
            self.touch_dir = (self.attacker[0] - self.fx,
                              self.attacker[1] - self.fy)

    def touch(self, dx, dy):
        """Тронуть муху с заданной стороны (в мировых координатах)."""
        self.touch_dir = (dx, dy)
        self._touched = True

    def touch_body(self, fx, fy):
        """Тронуть за конкретную часть ТЕЛА: fx вправо от мухи, fy вперёд.

        Обратное преобразование к fly_frame. Нужно потому, что муха
        поворачивается: касание в мировых координатах попадает то в левый
        бок, то в правый, и проверить сомато­топию так нельзя.
        """
        if self.fdir == NORTH:
            self.touch(fx, fy)
        elif self.fdir == SOUTH:
            self.touch(-fx, -fy)
        elif self.fdir == EAST:
            self.touch(fy, -fx)
        else:
            self.touch(-fy, fx)

    def displacement_towards(self, target, since=0):
        """Насколько муха сместилась в сторону точки, в клетках."""
        if target is None or len(self.path) <= since + 1:
            return 0.0
        x0, y0 = self.path[since]
        x1, y1 = self.path[-1]
        tx, ty = target
        d0 = math.hypot(tx - x0, ty - y0)
        d1 = math.hypot(tx - x1, ty - y1)
        return d0 - d1
