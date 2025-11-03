"""
Defines the types of problems the system can solve.
"""
from enum import Enum

class ProblemType(Enum):
    """Types de problèmes supportés"""
    OPTIMIZATION = "optimization"
    RL_CONTROL = "rl_control"
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"
