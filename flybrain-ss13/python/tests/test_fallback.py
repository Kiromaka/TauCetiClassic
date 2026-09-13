"""Резервный контур управления, когда нисходящие нейроны молчат.

На настоящем коннектоне до калибровки моторика может не отвечать вообще.
Без резервного контура муха в этом случае уходит строго на север и толкает
первую же стену до конца раунда, и по поведению не видно, что мозг не
работает. Тесты сторожат оба свойства: что муха всё-таки едет куда-то ещё
и что источник решения честно назван.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flybrain.decoders import EAST, NORTH, SOUTH, WEST, MotorDecoder  # noqa: E402

KEYS = ("escape", "forward", "backward", "turn_left", "turn_right",
        "groom", "threat", "feed", "speed")
SILENT = {k: 0.0 for k in KEYS}


def test_silent_brain_no_longer_locks_heading_north():
    """Регрессия на «идёт только наверх»."""
    dec = MotorDecoder()
    dirs = set()
    for t in range(60):
        a = dec.decode(SILENT, {"turn_left": 0, "turn_right": 0},
                       sensory={"vis_left": 0.32, "vis_right": 0.32},
                       total_spikes=3000 + t * 13)
        dirs.add(a.dir)
    assert len(dirs) > 1, f"курс так и не поменялся: {dirs}"
    assert dec.heading != 0.0


def test_fallback_can_be_switched_off():
    """Пуристу нужен режим «только мозг»: тогда молчание значит неподвижность."""
    dec = MotorDecoder(fallback=False, spontaneous=False)
    for t in range(40):
        a = dec.decode(SILENT, None, sensory={"vis_left": 0.9, "vis_right": 0.1},
                       total_spikes=5000)
    assert a.dir == NORTH and dec.heading == 0.0


def _opto(side_bright: str, warm: int = 30, hold: int = 6):
    """Прогреть на симметричной картинке, потом дать перекос."""
    dec = MotorDecoder()
    for _ in range(warm):
        dec.decode(SILENT, None, sensory={"vis_left": 0.4, "vis_right": 0.4},
                   total_spikes=0)
    prev = dec.heading
    bright = ({"vis_left": 0.2, "vis_right": 0.8} if side_bright == "right"
              else {"vis_left": 0.8, "vis_right": 0.2})
    turned = 0.0                      # накопленный поворот без заворачивания
    for _ in range(hold):
        dec.decode(SILENT, None, sensory=bright, total_spikes=0)
        turned += (dec.heading - prev + 180) % 360 - 180
        prev = dec.heading
    return dec, turned


def test_optomotor_turns_towards_brighter_eye():
    """Новый перекос зрительной активности разворачивает муху в свою сторону."""
    right, turned_r = _opto("right")
    left, turned_l = _opto("left")
    assert turned_r > 10, turned_r          # вправо = по часовой = курс растёт
    assert turned_l < -10, turned_l
    assert right.source == "оптомотор"


def test_optomotor_adapts_and_stops_spinning():
    """Постоянный перекос не должен крутить муху волчком.

    Зрительная система адаптируется к неизменной картинке; поворачиваться
    надо на ИЗМЕНЕНИЕ обстановки. Без адаптации муха с перекошенным полем
    зрения просто вращается на месте до конца раунда.
    """
    dec = MotorDecoder()
    total = 0.0
    prev = dec.heading
    for _ in range(120):
        dec.decode(SILENT, None, sensory={"vis_left": 0.2, "vis_right": 0.8},
                   total_spikes=0)
        total += abs((dec.heading - prev + 180) % 360 - 180)
        prev = dec.heading
    assert total < 360, f"намотала {total:.0f} градусов — это вращение на месте"


def test_bump_reflex_turns_the_fly_away():
    """Упёрлась в стену — через пару тиков должна развернуться."""
    dec = MotorDecoder()
    dec.arousal = 1.0                     # чтобы шаг точно был скомандован
    sources, headings = [], []
    for t in range(10):
        a = dec.decode(SILENT, None, sensory={"vis_left": 0.32, "vis_right": 0.32},
                       contact=0.9, total_spikes=2000)
        sources.append(a.source)
        headings.append(dec.heading)
    assert "удар о препятствие" in sources, sources
    spread = max(abs((b - a + 180) % 360 - 180)
                 for a in headings for b in headings)
    assert spread > 45, f"разворота от стены не случилось, разброс {spread:.0f}"


def test_bump_reflex_keeps_one_direction_for_a_while():
    """В углу нельзя качаться между двумя стенами — надо дожимать одну сторону."""
    dec = MotorDecoder()
    turns = []
    for t in range(30):
        dec.arousal = 1.0                 # муха всё время пытается идти
        before = dec.heading
        a = dec.decode(SILENT, None, sensory={"vis_left": 0.32, "vis_right": 0.32},
                       contact=0.9, total_spikes=2000)
        if a.source == "удар о препятствие":
            turns.append((dec.heading - before + 180) % 360 - 180)
    assert len(turns) >= 3, turns
    same = sum(1 for a, b in zip(turns, turns[1:]) if (a > 0) == (b > 0))
    assert same >= 1, f"направление разворота чередуется каждый раз: {turns}"


def test_source_is_reported_honestly():
    dec = MotorDecoder()
    for _ in range(20):                    # адаптация к симметричной картинке
        dec.decode(SILENT, None, sensory={"vis_left": 0.4, "vis_right": 0.4},
                   total_spikes=0)
    a = dec.decode(SILENT, None, sensory={"vis_left": 0.9, "vis_right": 0.1},
                   total_spikes=0)
    assert a.source == "оптомотор", a.source
    assert a.dbg["source"] == "оптомотор"
    assert a.dbg["brain_silent_for"] >= 1

    live = {**SILENT, "turn_right": 90.0, "forward": 30.0}
    for _ in range(30):
        b = dec.decode(live, {"turn_left": 1, "turn_right": 9},
                       sensory={"vis_left": 0.9, "vis_right": 0.1},
                       total_spikes=100)
    assert b.source in ("мозг", "шум DN"), b.source
    assert b.dbg["brain_silent_for"] == 0


def test_brain_signal_wins_over_fallback():
    """Если моторика заговорила, резервный контур обязан отойти в сторону."""
    dec = MotorDecoder()
    for _ in range(40):                    # набираем базовую линию
        dec.decode({**SILENT, "turn_left": 5.0, "turn_right": 5.0}, None,
                   sensory={"vis_left": 0.2, "vis_right": 0.8}, total_spikes=0)
    h0 = dec.heading
    for _ in range(8):                     # мозг тянет ВЛЕВО
        dec.decode({**SILENT, "turn_left": 120.0, "turn_right": 5.0}, None,
                   sensory={"vis_left": 0.2, "vis_right": 0.8}, total_spikes=0)
    assert dec.source == "мозг"
    assert (h0 - dec.heading) % 360 > 20, "поворот пошёл не туда, куда тянул мозг"


# ---------------------------------------------------------------------------
# Самопроверка рабочей точки на старте
# ---------------------------------------------------------------------------
def _agent_maker(cx, base_cfg):
    import copy
    from flybrain.agent import BrainPool
    pool = BrainPool(base_cfg, cx)

    def make(ws):
        from flybrain.agent import FlyAgent
        c = copy.copy(base_cfg.runtime)
        c.weight_scale = ws
        cfg2 = copy.copy(base_cfg)
        cfg2.runtime = c
        return FlyAgent("probe", cfg2, pool.cx, pool.pops, pool.hex, seed=3)
    return make


def test_dead_network_is_detected_and_repaired():
    """Главная ловушка настройки: при слабых весах сеть не спайкует, при
    сильных её давит гомеостаз, и снаружи оба случая выглядят одинаково."""
    from flybrain.autotune import autotune, probe_point
    from flybrain.config import Config
    from flybrain.connectome import make_synthetic

    cfg = Config()
    cfg.runtime.brain_ms_per_tick = 20.0
    cfg.runtime.background_hz = 0.0
    make = _agent_maker(make_synthetic(4000, 30, seed=9), cfg)

    dead = probe_point(lambda: make(0.0), ticks=4, weight_scale=0.0)
    assert dead.dead and not dead.ok, dead

    ws, seen = autotune(make, current=0.0, target_hz=2.0,
                        ladder=[0.5, 1.0, 2.0], ticks=4, log=lambda *_: None)
    assert ws != 0.0, (ws, seen)
    assert any(p.ok for p in seen), seen


def test_good_point_is_left_alone():
    """Если точка рабочая, перебор не запускается: это лишние секунды на
    старте и лишняя возможность что-нибудь сломать."""
    from flybrain.autotune import autotune
    from flybrain.config import Config
    from flybrain.connectome import make_synthetic

    cfg = Config()
    cfg.runtime.brain_ms_per_tick = 20.0
    make = _agent_maker(make_synthetic(4000, 30, seed=9), cfg)
    ws, seen = autotune(make, current=1.0, target_hz=2.0,
                        ladder=[0.5, 2.0, 4.0], ticks=4, log=lambda *_: None)
    if seen[0].ok:
        assert ws == 1.0 and len(seen) == 1, (ws, seen)


# ---------------------------------------------------------------------------
# Пищевое поведение
# ---------------------------------------------------------------------------
def test_hungry_fly_walks_towards_the_food():
    """Голод был виден на дашборде, но на поведение не влиял: муха с нулём
    калорий стояла рядом с булкой и не шла к ней."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arena import Arena
    from flybrain.agent import BrainPool
    from flybrain.config import Config
    from flybrain.connectome import make_synthetic

    cfg = Config()
    cfg.runtime.brain_ms_per_tick = 25.0
    pool = BrainPool(cfg, make_synthetic(5000, 30, seed=12))

    def run(nutrition):
        a = Arena(seed=3, clutter=6)
        a.nutrition = nutrition
        a.food = (a.fx, a.fy + 5)          # в пределах обоняния
        ag = pool.get(f"eat{nutrition}")
        # Прогрев: декодеру нужно снять постоянный перекос рулёжки, иначе
        # первые секунды муха едет по кругу. В игре это первые шесть секунд
        # после вселения.
        for _ in range(30):
            ag.step(a.percept(-1))
        for t in range(70):
            a.apply(ag.step(a.percept(t)))
        pool.drop(f"eat{nutrition}")
        return a.eaten

    hungry = run(40.0)       # голодает
    full = run(520.0)        # сыта
    assert hungry >= 3, hungry           # дошла и доела
    assert full == 0, full               # сытую еда не интересует


def test_tarsal_contact_triggers_eating_without_the_feed_population():
    """В части выгрузок FlyWire пищевых нисходящих нет вовсе. Тогда работает
    вторая дуга: вкус лапками плюс голод — рефлекс вытягивания хоботка."""
    from flybrain.decoders import MotorDecoder, COMMANDS, AUX
    keys = list(COMMANDS) + list(AUX)
    dec = MotorDecoder(spontaneous=False)
    rates = {k: 1.0 for k in keys}
    rates.pop("feed")                      # популяция не нашлась
    spikes = {k: 3 for k in rates}
    for _ in range(40):
        dec.decode(rates, spikes, hunger=0.0, taste=0.0)
    acts = [dec.decode(rates, spikes, hunger=0.95, taste=1.0).act
            for _ in range(20)]
    assert "eat" in acts, acts
    # но не каждый тик: одно вытягивание хоботка длиннее одного тика игры
    assert acts.count("eat") <= 7, acts.count("eat")


def test_sated_fly_does_not_eat_on_contact():
    from flybrain.decoders import MotorDecoder, COMMANDS, AUX
    keys = list(COMMANDS) + list(AUX)
    dec = MotorDecoder(spontaneous=False)
    rates = {k: 1.0 for k in keys}
    rates.pop("feed")
    spikes = {k: 3 for k in rates}
    for _ in range(40):
        dec.decode(rates, spikes)
    acts = [dec.decode(rates, spikes, hunger=0.05, taste=1.0).act
            for _ in range(20)]
    assert "eat" not in acts, acts


def test_chemotaxis_turns_towards_the_smell():
    from flybrain.decoders import MotorDecoder, COMMANDS, AUX
    keys = list(COMMANDS) + list(AUX)
    out = {}
    for label, bearing in (("справа", 90.0), ("слева", 270.0)):
        dec = MotorDecoder(spontaneous=False, fallback=False)
        rates = {k: 0.0 for k in keys}
        for _ in range(30):
            dec.decode(rates, None)
        h0 = dec.heading
        for _ in range(6):
            dec.decode(rates, None, food_bearing=bearing,
                       hunger=1.0, food_near=0.9)
        out[label] = ((dec.heading - h0 + 180) % 360) - 180
    assert out["справа"] > 15, out
    assert out["слева"] < -15, out


def test_arena_feeding_sequence_actually_consumes_food():
    """Подбор и поедание должны доводиться до конца, а не зацикливаться."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arena import Arena
    a = Arena(seed=1, clutter=0)
    a.food = (a.fx, a.fy)
    for _ in range(8):
        a.apply({"dir": a.fdir, "move": 0, "act": "eat"})
    assert a.eaten >= 3, a.eaten
    assert a.food is None and not a.food_held


def test_pain_reaches_the_escape_neuron():
    """Дуга «ноцицепция -> гигантский нейрон» должна работать целиком:
    удар в игре -> канал nocic -> DNp01 -> побег. Раньше на синтетике не
    было восходящих нейронов вовсе, канал боли не находил ни одного, и
    побег не мог сработать в принципе — а поведенческая проба по удару
    мерила случайную прогулку и объявляла её работающей."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arena import Arena
    from flybrain.agent import BrainPool
    from flybrain.config import Config
    from flybrain.connectome import make_synthetic

    cfg = Config()
    cfg.runtime.brain_ms_per_tick = 40.0
    cfg.runtime.dt = 1.0
    pool = BrainPool(cfg, make_synthetic(20000, 108, params=cfg.model))
    ag = pool.get("hurt")
    assert len(ag.pops.sensory["nocic"]) > 0, "канал боли пуст"

    a = Arena(seed=2, clutter=6)
    a.attacker = (a.fx + 2, a.fy)
    for _ in range(40):
        ag.step(a.percept(-1))
    quiet = ag.decoder._last_z["escape"]
    a.hit(0.9)
    states = []
    for t in range(8):
        out = ag.step(a.percept(100 + t))
        a.apply(out)
        states.append(out["dbg"]["state"])
    assert ag.decoder._last_z["escape"] > quiet + 1.0 or "escape" in states[0]
    assert "escape" in states, states
    assert states[-1] not in ("escape",), "побег должен заканчиваться"
