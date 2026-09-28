"""
Feedback Instruments 33-936S Sensor Noise & Stiction Model.
PCI-1711 DAQ Card Interface Simulation.

Features:
1. Optical Encoder Uniform Sensor Noise:
   - theta_noisy = theta + U(-0.5 deg, +0.5 deg)
   - x_noisy = x + U(-0.2 mm, +0.2 mm)
2. Optical Encoder Quantization (PCI-1711 quadrature counters):
   - Cart encoder: e.g. 4096 counts/m
   - Pendulum encoder: e.g. 4096 counts/rev (2*pi/4096 rad)
3. Asymmetric Dead-Zone / Stiction (Actuator & Cart interface):
   - Voltage dead-zone: +0.1V to +0.2V (forward), -0.2V to -0.1V (reverse)
   - Translated to force: F_stiction = K_amp * V_deadzone
   - Stiction/Coulomb friction opposing velocity when static or breaking breakaway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class SensorNoiseConfig:
    """Configuration parameters for DAQ sensor noise and stiction."""
    # Encoder uniform noise bounds
    theta_noise_deg: float = 0.5        # +/- 0.5 degrees
    x_noise_mm: float = 0.2             # +/- 0.2 mm

    # DAQ Optical Encoder Quantization
    enable_quantization: bool = True
    cart_counts_per_meter: float = 4096.0       # Cart linear encoder resolution
    pendulum_counts_per_rev: float = 4096.0     # Pendulum rotary encoder resolution

    # Asymmetric dead-zone (PCI-1711 D/A output voltage)
    enable_stiction: bool = True
    v_deadzone_positive: float = 0.15   # Volts (+0.1V to +0.2V)
    v_deadzone_negative: float = -0.12  # Volts (-0.1V to -0.2V)
    v_to_force_gain: float = 2.0        # N/V (yielding +/-20N at +/-10V)

    # Static / Breakaway friction force
    f_coulomb: float = 0.35             # Dynamic Coulomb friction [N]
    v_stiction_threshold: float = 0.005 # Velocity threshold for stiction lock [m/s]
    seed: int | None = None             # Random seed for reproducibility


class SensorNoiseModel:
    """
    Simulates PCI-1711 DAQ measurement noise and actuator stiction / dead-zone.
    """

    def __init__(self, config: SensorNoiseConfig | None = None) -> None:
        self.config = config or SensorNoiseConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Convert noise specifications to SI units
        self.theta_noise_rad = np.deg2rad(self.config.theta_noise_deg)
        self.x_noise_m = self.config.x_noise_mm * 1e-3

        # Quantization step sizes
        self.x_quant_step = 1.0 / self.config.cart_counts_per_meter
        self.theta_quant_step = (2.0 * np.pi) / self.config.pendulum_counts_per_rev

    def seed(self, seed: int) -> None:
        """Seed the random number generator."""
        self.rng = np.random.default_rng(seed)

    def measure(
        self, true_x: float, true_theta: float
    ) -> Tuple[float, float]:
        """
        Injects uniform optical encoder noise and DAQ quantization:
        theta_noisy = theta + U(-0.5 deg, +0.5 deg)
        x_noisy = x + U(-0.2 mm, +0.2 mm)
        Returns:
            (x_measured, theta_measured)
        """
        # 1. Inject uniform noise
        noise_x = self.rng.uniform(-self.x_noise_m, self.x_noise_m)
        noise_theta = self.rng.uniform(-self.theta_noise_rad, self.theta_noise_rad)

        x_noisy = true_x + noise_x
        theta_noisy = true_theta + noise_theta

        # 2. Apply encoder quantization if enabled
        if self.config.enable_quantization:
            x_noisy = np.round(x_noisy / self.x_quant_step) * self.x_quant_step
            theta_noisy = np.round(theta_noisy / self.theta_quant_step) * self.theta_quant_step

        return float(x_noisy), float(theta_noisy)

    def apply_actuator_deadzone_and_stiction(
        self, commanded_force: float, current_cart_velocity: float
    ) -> float:
        """
        Simulates asymmetric dead-zone (in volts) and stiction friction on the cart.
        - Translates commanded force to motor voltage: V = F / K_amp
        - Applies asymmetric deadzone [-V_neg, +V_pos]
        - If cart velocity is near zero (|x_dot| < v_thresh) and net force < breakaway force,
          cart remains locked by stiction.
        """
        if not self.config.enable_stiction:
            return commanded_force

        k_amp = self.config.v_to_force_gain
        v_cmd = commanded_force / k_amp

        # Asymmetric deadzone in voltage
        v_pos = self.config.v_deadzone_positive
        v_neg = self.config.v_deadzone_negative  # negative value e.g. -0.12V

        if v_neg <= v_cmd <= v_pos:
            # Within deadband
            effective_v = 0.0
        elif v_cmd > v_pos:
            effective_v = v_cmd - v_pos
        else:
            effective_v = v_cmd - v_neg

        effective_force = effective_v * k_amp

        # Stiction / Karnopp model near zero cart velocity
        if abs(current_cart_velocity) < self.config.v_stiction_threshold:
            breakaway_force = max(abs(v_pos), abs(v_neg)) * k_amp + self.config.f_coulomb
            if abs(effective_force) < breakaway_force:
                # Motor cannot overcome static friction
                return 0.0
            else:
                # Breakaway achieved: subtract Coulomb friction
                return effective_force - np.sign(effective_force) * self.config.f_coulomb
        else:
            # Dynamic friction opposes velocity
            coulomb_drag = self.config.f_coulomb * np.sign(current_cart_velocity)
            return effective_force - coulomb_drag
