"""
Lyapunov-based weight adaptation for continuous-time LSTM observers.
"""

from .jacobian_engine import (
    LSTMWeightLayout,
    LSTMCache,
    LyapunovAdaptationLaw,
    lstm_forward,
    gate_sensitivities,
    phi_jacobian,
    phi_jacobian_transpose_vec,
    smooth_projection,
    sigmoid,
)

__all__ = [
    "LSTMWeightLayout",
    "LSTMCache",
    "LyapunovAdaptationLaw",
    "lstm_forward",
    "gate_sensitivities",
    "phi_jacobian",
    "phi_jacobian_transpose_vec",
    "smooth_projection",
    "sigmoid",
]
