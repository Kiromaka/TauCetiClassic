"""Цифровой двойник мозга дрозофилы, подключённый к SS13 (TauCetiClassic).

    from flybrain import Config, BrainPool
    from flybrain.server import serve
"""
from .config import Config, ModelParams, RuntimeParams, ServerParams  # noqa: F401
from .connectome import Connectome, build, make_synthetic              # noqa: F401
from .lif import LIFEngine, TickResult                                 # noqa: F401
from .agent import BrainPool, FlyAgent                                 # noqa: F401
from .encoders import Percept, SensoryEncoder                          # noqa: F401
from .decoders import MotorDecoder, Action                             # noqa: F401

__version__ = "1.0.0"
__all__ = ["Config", "BrainPool", "FlyAgent", "LIFEngine", "Connectome",
           "Percept", "SensoryEncoder", "MotorDecoder", "Action",
           "build", "make_synthetic"]
