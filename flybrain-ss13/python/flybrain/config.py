"""Конфигурация модели мозга и рантайма.

Параметры LIF взяты из Shiu et al., Nature 2024 (philshiu/Drosophila_brain_model),
чтобы динамика совпадала с опубликованной моделью.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List


# --------------------------------------------------------------------------
# Модель нейрона
# --------------------------------------------------------------------------
@dataclass
class ModelParams:
    """LIF-параметры Shiu et al. 2024. Всё в мВ и мс."""

    v_rest: float = -52.0        # v_0,  потенциал покоя
    v_reset: float = -52.0       # v_rst
    v_threshold: float = -45.0   # v_th
    tau_membrane: float = 20.0   # t_mbr
    tau_synapse: float = 5.0     # tau
    t_refractory: float = 2.2    # t_rfc
    t_delay: float = 1.8         # t_dly
    w_synapse: float = 0.275     # мВ на одну синаптическую контактную точку

    # Глобальный gain. 1.0 = ровно по статье. Проекты, гоняющие коннектом
    # в реальном времени, обычно калибруют его в 0.6-0.8, иначе сеть
    # уходит в эпилептический разгон при постоянной сенсорной накачке.
    gain: float = 0.65

    # Знак синапса по предсказанному нейромедиатору пресинаптического нейрона.
    # DA/SER/OCT — нейромодуляторы; в модели Shiu быстрого синаптического
    # эффекта им не приписывают, поэтому 0. Поставьте 1.0, если хотите,
    # чтобы моноамины работали как возбуждающие.
    nt_sign: Dict[str, float] = field(default_factory=lambda: {
        "ACH": +1.0, "ACETYLCHOLINE": +1.0,
        "GABA": -1.0,
        "GLUT": -1.0, "GLUTAMATE": -1.0,
        "DA": 0.0, "DOPAMINE": 0.0,
        "SER": 0.0, "SEROTONIN": 0.0,
        "OCT": 0.0, "OCTOPAMINE": 0.0,
        "UNK": 0.0, "": 0.0,
    })

    # Минимальное число синапсов, чтобы считать связь существующей.
    # Codex-выгрузка connections.csv уже отфильтрована по >=5.
    min_syn_count: int = 5


# --------------------------------------------------------------------------
# Рантайм
# --------------------------------------------------------------------------
@dataclass
class RuntimeParams:
    # Множитель весов, применяемый при загрузке. model.gain запекается в
    # .npz на этапе сборки, а этот работает в рантайме — им и подбирают
    # рабочую точку, не пересобирая коннектом по полчаса.
    weight_scale: float = 1.0

    dt: float = 0.5              # шаг интегрирования, мс. 0.1 = как в статье
    brain_ms_per_tick: float = 50.0   # сколько мс мозгового времени на один тик игры
    backend: str = "auto"        # auto | numpy | torch
    device: str = "cpu"          # для torch: cpu | cuda
    dtype: str = "float32"

    # Если один тик считался дольше, чем этот бюджет (сек), движок сам
    # урежет brain_ms_per_tick. Муха начнёт жить в замедленной съёмке,
    # но сервер не ляжет.
    tick_budget_s: float = 0.18
    adaptive: bool = True

    # Пуассоновская накачка сенсорных нейронов: максимальная частота (Гц),
    # соответствующая сенсорному сигналу 1.0.
    sensory_max_hz: float = 250.0   # f_poi из статьи

    # Насколько сильно сенсорный сигнал 1.0 двигает синаптическую переменную g.
    # Порог минус покой = 7 мВ, поэтому при 14 мВ сигнал 0.5 садится ровно
    # на порог, а 1.0 уверенно гонит нейрон в спайки. Отсюда движок сам
    # считает вес виртуального афферента.
    input_drive_mv: float = 14.0

    # Глобальное тормозное управление усилением.
    # Настоящий коннектом при постоянной сенсорной накачке легко уходит в
    # разгон: у Shiu et al. модель работает короткими импульсами и такой
    # проблемы не видит, а в игре накачка идёт непрерывно. Один скалярный
    # тормозной ток, подстраиваемый под целевую среднюю частоту, держит сеть
    # в рабочем режиме и не трогает ОТНОСИТЕЛЬНУЮ модуляцию, по которой
    # и принимает решения декодер. Это добавка сверх статьи; homeostasis=False
    # возвращает поведение ровно по статье.
    homeostasis: bool = True
    target_rate_hz: float = 2.0
    homeostasis_gain: float = 0.35      # мВ тормозного тока на 1 Гц перебора
    # Отпускать тормоз надо заметно быстрее, чем зажимать: иначе один
    # всплеск активности глушит сеть на десятки тиков и муха "слепнет".
    homeostasis_release: float = 1.5    # мВ на 1 Гц недобора
    homeostasis_max: float = 80.0

    # --- самопроверка на старте ------------------------------------------
    # Сервер меряет рабочую точку до первого игрока: мёртвая сеть и сеть в
    # разгоне снаружи неотличимы, и то и другое выглядит как "DN молчат".
    startup_check: bool = True
    startup_ticks: int = 8
    autotune: bool = True            # чинить рабочую точку самому
    autotune_persist: bool = True    # и записывать найденное в конфиг

    # --- сетчатка -------------------------------------------------------
    # grid        — тайлы игры кладутся в сетчатку один в один (наивно:
    #               надвигание не кодируется вообще);
    # ground      — честная проекция на пол, углы места равномерны;
    # perspective — то же, но расстояние разложено логарифмически, чтобы
    #               дальние тайлы не оставались без единого омматидия.
    projection: str = "perspective"
    eye_height_tiles: float = 0.6    # высота глаза в тайлах (для ground)
    near_tiles: float = 0.75         # ближняя граница поля, тайлов
    # Поле зрения одного глаза и фронтальное перекрытие, градусов.
    # У дрозофилы глаз видит почти полусферу, вдвоём — почти круг.
    eye_fov_deg: float = 195.0
    eye_overlap_deg: float = 25.0

    # Фоновая активность (Гц на нейрон), чтобы сеть не была абсолютно мёртвой.
    # Вес фонового события — как у 5 синаптических контактов.
    background_hz: float = 3.0
    background_synapses: int = 5


# --------------------------------------------------------------------------
# Популяции нейронов: как их находить в аннотациях FlyWire
# --------------------------------------------------------------------------
@dataclass
class PopulationSpec:
    """Правило отбора нейронов.

    match_type   — regex по колонкам cell_type / hemibrain_type / consolidated type
    match_class  — regex по колонке class
    match_super  — regex по колонке super_class
    side         — 'left' | 'right' | None
    Все заданные условия объединяются по И.
    """
    match_type: str | None = None
    match_class: str | None = None
    match_super: str | None = None
    side: str | None = None
    # Для зрительных популяций: раскладывать ли ретинотопически
    retinotopic: bool = False
    # Куда этот сенсорный канал «должен» тянуть моторику. Используется,
    # только если несколько каналов схлопнулись в один набор нейронов и
    # их приходится разделять по связности.
    affinity: List[str] = field(default_factory=list)
    # Более широкий шаблон на случай, если по основному нашлось слишком
    # мало нейронов. Применяется с отчётом, молча ничего не подменяется.
    widen: str | None = None
    min_size: int = 0


def default_sensory_specs() -> Dict[str, PopulationSpec]:
    """Сенсорные входы. Порядок в списке важен — используется как приоритет
    при разрешении зрительной популяции (см. populations.resolve_visual)."""
    return {
        # -- зрение --------------------------------------------------------
        # super_class в свежих выгрузках включает optic, visual_projection и
        # visual_centrifugal; класс у части зрительных нейронов пустой,
        # поэтому по классу не фильтруем — ярус выбирается ниже по связности
        "vis_left": PopulationSpec(
            match_super=r"^(sensory|optic|visual_projection)$",
            side="left", retinotopic=True),
        "vis_right": PopulationSpec(
            match_super=r"^(sensory|optic|visual_projection)$",
            side="right", retinotopic=True),
        # -- обоняние ------------------------------------------------------
        "olf_attractive": PopulationSpec(
            match_type=r"^ORN_(DM1|DM4|VA2|DC2|VM7)", match_class=r"olfactory",
            affinity=["forward", "feed"]),
        "olf_aversive": PopulationSpec(
            match_type=r"^ORN_(DA2|DL5|V\b|VM3)|Or56a|geosmin", match_class=r"olfactory",
            affinity=["escape", "backward"]),
        "olf_co2": PopulationSpec(match_type=r"^ORN_V$|Gr21a|Gr63a", match_class=r"olfactory",
                                  affinity=["escape"]),
        # -- вкус ----------------------------------------------------------
        # В аннотациях FlyWire подтипов вкусовых нейронов может не быть
        # вовсе — тогда все три канала схлопнутся в один класс. Резолвер
        # разделит его по связности, опираясь на affinity.
        "gust_sugar": PopulationSpec(match_type=r"Gr5a|Gr64|sugar|sweet",
                                     match_class=r"gustatory",
                                     affinity=["feed", "forward"]),
        "gust_bitter": PopulationSpec(match_type=r"Gr66a|Gr33a|bitter",
                                      match_class=r"gustatory",
                                      affinity=["backward", "escape"]),
        "gust_water": PopulationSpec(match_type=r"ppk28|water", match_class=r"gustatory",
                                     affinity=["feed"]),
        # -- механорецепция ------------------------------------------------
        "wind_jo": PopulationSpec(match_type=r"^JO-(A|B|C|D|E|F)", match_class=r"mechanosensory"),
        "bristle": PopulationSpec(match_type=r"bristle|BM_|^BM", match_class=r"mechanosensory"),
        "mechano_any": PopulationSpec(match_class=r"mechanosensory"),
        # -- боль ----------------------------------------------------------
        # Отдельного класса ноцицепторов в аннотациях взрослой мухи нет:
        # у имаго повреждение тела приходит в мозг восходящими нейронами
        # из брюшной нервной цепочки. Их и берём.
        "nocic": PopulationSpec(match_super=r"ascending",
                                affinity=["escape", "backward"]),
        # -- голод ---------------------------------------------------------
        # Интероцепция: пептидергические нейроны состояния сытости.
        # Если таких типов в снапшоте нет, канал остаётся пустым, и голод
        # работает только как множитель к аппетитивным путям (см. encoders).
        "hunger": PopulationSpec(match_type=r"DH44|AKH|CRZ|[Hh]ugin|NPF|CN\b|Fdg",
                                 affinity=["feed", "forward"]),
        # Жажда. Ищем ISN/ITP/DH44-подобные интероцепторы осмотического
        # состояния. Если не нашлось — канал просто пустой, а сама жажда
        # всё равно открывает вкусовой канал воды: так у мухи и устроено.
        "thirst": PopulationSpec(match_type=r"ISN|ITP|DH44|SLP\b|\bDSK\b",
                                 affinity=["feed", "forward"]),
        # Запах брожения. Отдельной аннотации "этанол" в выгрузке нет,
        # поэтому берём тот же аппетитивный обонятельный путь, что и еда:
        # для дрозофилы брожение это и есть еда. Ярлык — гипотеза.
        "olf_ethanol": PopulationSpec(
            match_type=r"ORN_DM1|ORN_DM4|ORN_VA2|ORN_DP1m|Or42|Or92|Or59",
            match_class=r"olfactory", affinity=["feed", "forward"]),
        # -- температура ---------------------------------------------------
        "thermo_hot": PopulationSpec(match_type=r"\bHC\b|hot|warm", match_class=r"thermosensory",
                                     affinity=["escape", "backward"]),
        "thermo_cold": PopulationSpec(match_type=r"\bCC\b|cold|cool", match_class=r"thermosensory",
                                      affinity=["forward"]),
        "thermo_any": PopulationSpec(match_class=r"thermo|hygro"),
    }


def default_motor_specs() -> Dict[str, PopulationSpec]:
    """Нисходящие нейроны, с которых снимаем моторные команды.

    Роли по литературе:
      DNp01 (Giant Fiber)  — эскейп-тейкофф
      DNp09                — гейт передней ходьбы / замирание
      MDN  (moonwalker)    — задний ход
      DNa01, DNa02         — рулёжка (лево/право по стороне тела)
      DNg11                — груминг
      DNp02/DNp04/DNp11    — угроза, движения крыльями
      DNa08 / DNg13        — модуляция скорости
    """
    return {
        "escape":    PopulationSpec(match_type=r"^DNp01\b|Giant.?Fiber|^GF$", match_super=r"descending"),
        "forward":   PopulationSpec(match_type=r"^DNp09\b", match_super=r"descending"),
        "backward":  PopulationSpec(match_type=r"^MDN\b|moonwalker", match_super=r"descending"),
        "turn_left": PopulationSpec(match_type=r"^DNa0(1|2)\b", match_super=r"descending",
                                    side="left", widen=r"^DNa\d+", min_size=6),
        "turn_right": PopulationSpec(match_type=r"^DNa0(1|2)\b", match_super=r"descending",
                                     side="right", widen=r"^DNa\d+", min_size=6),
        "groom":     PopulationSpec(match_type=r"^DNg11\b", match_super=r"descending"),
        "threat":    PopulationSpec(match_type=r"^DNp(02|04|11)\b", match_super=r"descending"),
        # Расширять этот канал шаблоном "." нельзя: с match_super
        # "descending|motor" он тогда забирает ВСЮ нисходящую популяцию и
        # превращается из команды еды в индикатор общей активности —
        # муха начинает тянуться к еде на любой раздражитель.
        "feed":      PopulationSpec(match_type=r"^MN9\b|Fdg|proboscis|pharyn|rostrum|haustell",
                                    match_super=r"motor",
                                    widen=r"^MN\d+|motor", min_size=1),
        "speed":     PopulationSpec(match_type=r"^DNa08\b|^DNg13\b", match_super=r"descending"),

        # Популяционная рулёжка. Рулевой сигнал у мухи размазан по многим
        # нисходящим нейронам, а не сидит в DNa01/DNa02: см. работы про
        # population-based descending control. Именованных типов набирается
        # по три нейрона на сторону, и частота у них квантована ступенями
        # в полгерца — шумно. Асимметрия всей нисходящей популяции даёт
        # тот же по смыслу сигнал, но на сотнях нейронов.
        "dn_left":   PopulationSpec(match_super=r"descending", side="left"),
        "dn_right":  PopulationSpec(match_super=r"descending", side="right"),

        # общий уровень активности — из него вычитается общая мода
        "dn_all":    PopulationSpec(match_super=r"descending"),
    }


@dataclass
class ServerParams:
    host: str = "127.0.0.1"
    port: int = 5665
    secret: str = "change-me"
    # Обратный канал в BYOND (world/Topic). Нужен только для внеполосных
    # команд; основной цикл идёт в ответе на HTTP-запрос из DM.
    byond_host: str = "127.0.0.1"
    byond_port: int = 0          # 0 = обратный канал выключен
    viz: bool = True


# Готовые режимы. Отличаются только ценой шага интегрирования и тем,
# сколько мозгового времени проживает муха за один игровой тик.
PRESETS = {
    # экономный: много мух или слабая машина
    "fast":     {"dt": 1.0, "brain_ms_per_tick": 40.0},
    # по умолчанию: запас по бюджету примерно вчетверо
    "balanced": {"dt": 0.5, "brain_ms_per_tick": 50.0},
    # мозговое время идёт вровень с реальным: муха живёт не в замедленной
    # съёмке. Нужен быстрый процессор, около 200 мс счёта на тик
    "realtime": {"dt": 0.5, "brain_ms_per_tick": 200.0},
    # шаг как в статье Shiu et al.; для изучения динамики, не для игры
    "faithful": {"dt": 0.1, "brain_ms_per_tick": 50.0},
}


@dataclass
class Config:
    model: ModelParams = field(default_factory=ModelParams)
    runtime: RuntimeParams = field(default_factory=RuntimeParams)
    server: ServerParams = field(default_factory=ServerParams)
    data_dir: str = "data"
    connectome_file: str = "data/connectome.npz"
    annotations_file: str = "data/annotations.npz"
    sensory: Dict[str, PopulationSpec] = field(default_factory=default_sensory_specs)
    motor: Dict[str, PopulationSpec] = field(default_factory=default_motor_specs)
    loaded_from: str | None = None    # откуда прочитан конфиг, для отчёта

    # ------------------------------------------------------------------
    # откуда сам подберётся конфиг, если путь не задан явно
    AUTO_PATHS = ("flybrain.json", "config/flybrain.json",
                  "../flybrain.json")

    @classmethod
    def find_config(cls, path: str | None = None) -> str | None:
        """Найти конфиг. Раньше лежащий рядом flybrain.json просто
        игнорировался, если не передать --config, и правки в нём (тот же
        weight_scale) молча не применялись."""
        if path:
            return path if os.path.exists(path) else None
        env = os.environ.get("FLYBRAIN_CONFIG")
        if env and os.path.exists(env):
            return env
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for cand in cls.AUTO_PATHS:
            for base in (os.getcwd(), here):
                full = os.path.join(base, cand)
                if os.path.exists(full):
                    return full
        return None

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        path = cls.find_config(path)
        cfg.loaded_from = path
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            cfg = cls._merge(cfg, raw)
            cfg.loaded_from = path
        # переменные окружения имеют высший приоритет — удобно для docker
        if os.environ.get("FLYBRAIN_SECRET"):
            cfg.server.secret = os.environ["FLYBRAIN_SECRET"]
        if os.environ.get("FLYBRAIN_PORT"):
            cfg.server.port = int(os.environ["FLYBRAIN_PORT"])
        if os.environ.get("FLYBRAIN_HOST"):
            cfg.server.host = os.environ["FLYBRAIN_HOST"]
        if os.environ.get("FLYBRAIN_BACKEND"):
            cfg.runtime.backend = os.environ["FLYBRAIN_BACKEND"]
        if os.environ.get("FLYBRAIN_DEVICE"):
            cfg.runtime.device = os.environ["FLYBRAIN_DEVICE"]
        return cfg

    @staticmethod
    def _merge(cfg: "Config", raw: dict) -> "Config":
        for section in ("model", "runtime", "server"):
            if section in raw:
                obj = getattr(cfg, section)
                for k, v in raw[section].items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
        for key in ("data_dir", "connectome_file", "annotations_file"):
            if key in raw:
                setattr(cfg, key, raw[key])
        for section, target in (("sensory", cfg.sensory), ("motor", cfg.motor)):
            for name, spec in raw.get(section, {}).items():
                target[name] = PopulationSpec(**spec)
        return cfg

    def patch_file(self, updates: Dict[str, dict],
                   path: str | None = None) -> str:
        """Точечно записать несколько полей в конфиг, не трогая остальное.

        Нужно, чтобы подобранные значения применялись сами. Раньше скрипты
        печатали "пропишите weight_scale в flybrain.json", и ровно на этом
        шаге настройка и терялась: сервер перезапускали, а файл оставался
        прежним, и мозг работал на старой рабочей точке.
        """
        path = path or self.loaded_from
        if not path:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, "flybrain.json")
        raw: dict = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
            except (OSError, ValueError):
                raw = {}
        for section, fields in updates.items():
            raw.setdefault(section, {}).update(fields)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        self.loaded_from = path
        return path

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "model": asdict(self.model),
                "runtime": asdict(self.runtime),
                "server": asdict(self.server),
                "data_dir": self.data_dir,
                "connectome_file": self.connectome_file,
                "annotations_file": self.annotations_file,
                "sensory": {k: asdict(v) for k, v in self.sensory.items()},
                "motor": {k: asdict(v) for k, v in self.motor.items()},
            }, fh, indent=2, ensure_ascii=False)
