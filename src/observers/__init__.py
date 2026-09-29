"""
Lyapunov-based LSTM adaptive observers.
Branch: feature/shared-benchmark (Approaches A, B, C and D side by side)

- PIICLObserver: physics-structured observer with integral concurrent learning (Approach D)
- CLLbLSTMObserver: concurrent-learning Lb-LSTM (Approach C)
- PILSTMObserver: physics-informed Lb-LSTM (Approach B, structured Euler-Lagrange sub-networks)
- LbLSTMObserver: black-box Lb-LSTM (Approach A baseline, imported from its branch)
- PhysicsInformedLSTMObserver: earlier known-parameter Euler-Lagrange prototype
"""

from .blackbox_lstm import LbLSTMObserver, LbLSTMObserverConfig, ObserverDerivatives
from .physics_informed_lstm_observer import (
    PhysicsInformedLSTMObserver,
    PhysicsInformedLSTMConfig,
)
from .pilstm_network import PILSTMNetwork, PILSTMLayout, PILSTMCache
from .pilstm_observer import PILSTMObserver, PILSTMObserverConfig
from .el_linear_model import LinearELModel, LinearELLayout
from .pi_icl_observer import PIICLObserver, PIICLObserverConfig
from .cl_lstm_observer import CLLbLSTMObserver, CLLbLSTMObserverConfig

__all__ = [
    "CLLbLSTMObserver",
    "CLLbLSTMObserverConfig",
    "PIICLObserver",
    "PIICLObserverConfig",
    "LinearELModel",
    "LinearELLayout",
    "PILSTMObserver",
    "PILSTMObserverConfig",
    "PILSTMNetwork",
    "PILSTMLayout",
    "PILSTMCache",
    "LbLSTMObserver",
    "LbLSTMObserverConfig",
    "ObserverDerivatives",
    "PhysicsInformedLSTMObserver",
    "PhysicsInformedLSTMConfig",
]
