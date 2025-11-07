"""
Competitive MARL - 3D Pursuit-Evasion

Clean, modular competitive multi-agent reinforcement learning system
optimized for Kaggle GPU training.
"""

__version__ = "1.0.0"
__author__ = "PIDRL Team"

from .environment.pursuit_evasion_3d import CompetitivePursuitEvasion3D, get_target_observation
from .agents import PursuerAgent, EvaderAgent
from .config import get_config

__all__ = [
    "CompetitivePursuitEvasion3D",
    "get_target_observation",
    "PursuerAgent",
    "EvaderAgent",
    "get_config",
]
