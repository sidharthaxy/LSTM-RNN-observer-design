"""
Open-loop plant rollouts for observer validation and system identification.

Unlike `benchmark_runner.run_plant_simulation`, which only supports the built-in
excitation modes and records the post-stiction force, this helper accepts an
arbitrary input signal u(t) and records both the commanded force (what a real
controller knows) and the net force the cart actually receives after the
PCI-1711 dead-zone / stiction model, together with the true accelerations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from src.plant.pendulum_plant import PendulumPlant, PendulumParameters, PlantState
from src.plant.sensor_noise import SensorNoiseModel, SensorNoiseConfig


@dataclass
class OpenLoopTrajectory:
    """Sampled open-loop trajectory of the cart-pendulum rig."""
    t: np.ndarray              # Shape (N,)
    state: np.ndarray          # Shape (N, 4): true [x, x_dot, theta, theta_dot]
    accel: np.ndarray          # Shape (N, 2): true [x_ddot, theta_ddot] under the net force
    measurements: np.ndarray   # Shape (N, 2): noisy, quantized encoder [x, theta]
    u_cmd: np.ndarray          # Shape (N,): commanded force [N]
    u_net: np.ndarray          # Shape (N,): force after dead-zone / stiction [N]

    @property
    def positions(self) -> np.ndarray:
        """True generalized coordinates x1 = [x, theta], shape (N, 2)."""
        return self.state[:, [0, 2]]

    @property
    def velocities(self) -> np.ndarray:
        """True generalized velocities x2 = [x_dot, theta_dot], shape (N, 2)."""
        return self.state[:, [1, 3]]


def simulate_open_loop(
    u_fn: Callable[[float], float],
    t_final: float,
    dt: float = 0.001,
    initial_state: PlantState | np.ndarray | None = None,
    plant_params: PendulumParameters | None = None,
    noise_config: SensorNoiseConfig | None = None,
    apply_actuator_effects: bool = True,
    seed: int | None = 42,
) -> OpenLoopTrajectory:
    """
    Simulates the Feedback 33-936S rig under the open-loop force u(t) using RK4.

    Every sample k records the state at t_k, the encoder measurement of that state,
    and the zero-order-held force applied over [t_k, t_k + dt).
    """
    plant = PendulumPlant(plant_params)
    plant.reset(initial_state)
    noise_model = SensorNoiseModel(noise_config)
    if seed is not None:
        noise_model.seed(seed)

    t_arr = np.arange(0.0, t_final, dt)
    n_steps = len(t_arr)
    state_hist = np.zeros((n_steps, 4), dtype=np.float64)
    accel_hist = np.zeros((n_steps, 2), dtype=np.float64)
    meas_hist = np.zeros((n_steps, 2), dtype=np.float64)
    u_cmd_hist = np.zeros(n_steps, dtype=np.float64)
    u_net_hist = np.zeros(n_steps, dtype=np.float64)

    for k, t_val in enumerate(t_arr):
        state = plant.state.to_array()
        state_hist[k] = state
        meas_hist[k] = noise_model.measure(state[0], state[2])

        u_cmd = float(u_fn(float(t_val)))
        if apply_actuator_effects:
            u_net = noise_model.apply_actuator_deadzone_and_stiction(u_cmd, state[1])
        else:
            u_net = u_cmd
        u_cmd_hist[k] = u_cmd
        u_net_hist[k] = u_net
        accel_hist[k] = plant.forward_dynamics(state, u_net)

        plant.step(u_net, dt, integrator="rk4")

    return OpenLoopTrajectory(
        t=t_arr,
        state=state_hist,
        accel=accel_hist,
        measurements=meas_hist,
        u_cmd=u_cmd_hist,
        u_net=u_net_hist,
    )
