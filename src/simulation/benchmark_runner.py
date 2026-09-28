"""
Benchmark Simulation Testbed for the Feedback 33-936S Pendulum Rig.
Runs closed-loop / open-loop experiments, injects realistic PCI-1711 DAQ noise,
and benchmarks state/velocity estimators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional
import numpy as np
import scipy.linalg

from src.plant.pendulum_plant import PendulumPlant, PendulumParameters, PlantState
from src.plant.sensor_noise import SensorNoiseModel, SensorNoiseConfig
from src.baselines.classical_estimators import (
    DirtyDerivativeFilter,
    ButterworthDifferentiator,
    ContinuousDiscreteEKF,
    ContinuousShallowRNNObserver,
    EKFConfig,
    RNNConfig,
)
from src.utils.metrics import evaluate_estimation_performance, EstimationReport


@dataclass
class SimulationConfig:
    """Simulation run configuration."""
    t_final: float = 10.0           # Total simulation time [s]
    dt: float = 0.001              # Sampling period [s] (1 kHz matches PCI-1711 DAQ)
    initial_state: PlantState = field(default_factory=lambda: PlantState(x=0.0, x_dot=0.0, theta=0.05, theta_dot=0.0))
    excitation_mode: str = "lqr_stabilize"  # "lqr_stabilize", "chirp", "multisine", "step"
    integrator: str = "rk4"
    seed: int = 42


@dataclass
class SimulationResult:
    """Container for benchmark trajectory histories."""
    t: np.ndarray
    state_true: np.ndarray         # Shape (N, 4): [x, x_dot, theta, theta_dot]
    measurements: np.ndarray       # Shape (N, 2): [x_noisy, theta_noisy]
    force_applied: np.ndarray      # Shape (N,)
    estimates: Dict[str, np.ndarray] # Key -> Shape (N, 4) state estimate
    reports: Dict[str, EstimationReport]


def compute_lqr_gain(plant: PendulumPlant) -> np.ndarray:
    """
    Computes optimal infinite-horizon LQR gain K for stabilizing upright pendulum (theta = 0).
    Solves continuous Algebraic Riccati Equation (CARE) A^T P + P A - P B R^-1 B^T P + Q = 0.
    """
    eq_state = np.zeros(4, dtype=np.float64)
    A, B = plant.analytical_jacobian(eq_state, 0.0)

    # State weights: penalize cart position, cart velocity, angle, angular velocity
    Q = np.diag([10.0, 1.0, 100.0, 5.0])
    # Control effort weight
    R = np.array([[0.01]])

    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ P)
    return K.flatten()  # 1D array of shape (4,)


def generate_excitation_force(
    t: float,
    mode: str = "chirp",
    cart_pos: float = 0.0,
    lqr_gain: np.ndarray | None = None,
    state: np.ndarray | None = None,
) -> float:
    """
    Generates excitation force input for system identification & estimation benchmarks.
    """
    if mode == "chirp":
        # Frequency sweep from 0.2 Hz to 4.0 Hz
        f0, f1, t1 = 0.2, 4.0, 10.0
        inst_f = f0 + (f1 - f0) * (t / t1)
        return float(5.0 * np.sin(2.0 * np.pi * inst_f * t))

    elif mode == "multisine":
        # Sum of orthogonal sinusoids (persistent excitation)
        omega = [1.2, 3.4, 7.8, 12.5]
        amps = [3.0, 2.0, 1.5, 1.0]
        return float(sum(a * np.sin(w * t) for a, w in zip(amps, omega)))

    elif mode == "step":
        # Square pulse sequence
        return 4.0 if (int(t) % 2 == 0) else -4.0

    elif mode == "lqr_stabilize":
        # Closed-loop stabilization around upright (theta=0) + reference cart tracking
        if lqr_gain is None or state is None:
            return 0.0
        # Reference cart trajectory: 0.15 * sin(1.5 * t)
        x_ref = 0.15 * np.sin(1.5 * t)
        x_dot_ref = 0.15 * 1.5 * np.cos(1.5 * t)
        ref_state = np.array([x_ref, x_dot_ref, 0.0, 0.0], dtype=np.float64)

        error_state = state - ref_state
        # u = -K * e
        u_lqr = -float(np.dot(lqr_gain, error_state))
        return float(np.clip(u_lqr, -20.0, 20.0))

    else:
        return 0.0


def run_plant_simulation(
    sim_config: SimulationConfig | None = None,
    plant_params: PendulumParameters | None = None,
    noise_config: SensorNoiseConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PendulumPlant]:
    """
    Executes forward simulation of the true physical plant with PCI-1711 DAQ noise.
    Returns:
        (t, state_history, measurement_history, force_history, plant_instance)
    """
    cfg = sim_config or SimulationConfig()
    plant = PendulumPlant(plant_params)
    noise_model = SensorNoiseModel(noise_config)
    if cfg.seed is not None:
        noise_model.seed(cfg.seed)

    plant.reset(cfg.initial_state)
    lqr_gain = compute_lqr_gain(plant) if cfg.excitation_mode == "lqr_stabilize" else None

    t_arr = np.arange(0.0, cfg.t_final, cfg.dt)
    n_steps = len(t_arr)

    state_hist = np.zeros((n_steps, 4), dtype=np.float64)
    meas_hist = np.zeros((n_steps, 2), dtype=np.float64)
    force_hist = np.zeros(n_steps, dtype=np.float64)

    for i, t_val in enumerate(t_arr):
        curr_state = plant.state.to_array()
        state_hist[i] = curr_state

        # Optical encoder measurement
        x_meas, theta_meas = noise_model.measure(curr_state[0], curr_state[2])
        meas_hist[i] = [x_meas, theta_meas]

        # Generate control / excitation force
        cmd_force = generate_excitation_force(
            t_val,
            mode=cfg.excitation_mode,
            cart_pos=curr_state[0],
            lqr_gain=lqr_gain,
            state=curr_state,
        )

        # Apply stiction / asymmetric deadband
        net_force = noise_model.apply_actuator_deadzone_and_stiction(cmd_force, curr_state[1])
        force_hist[i] = net_force

        # Forward integration step
        plant.step(net_force, cfg.dt, integrator=cfg.integrator)

    return t_arr, state_hist, meas_hist, force_hist, plant


def run_benchmark_comparison(
    sim_config: SimulationConfig | None = None,
    custom_observers: Dict[str, Any] | None = None,
) -> SimulationResult:
    """
    Executes standard benchmark comparison on Feedback 33-936S rig.
    Compares:
    1. Dirty Derivative Filter (1st order)
    2. Butterworth Filter Differentiator (2nd order from Feedback manual)
    3. Continuous-Discrete EKF
    4. Continuous Shallow Dynamic RNN Observer (Dinh et al., 2014)
    + Any user-supplied custom observers (such as Lb-LSTM architectures).
    """
    cfg = sim_config or SimulationConfig()
    t_arr, state_hist, meas_hist, force_hist, plant = run_plant_simulation(cfg)
    n_steps = len(t_arr)
    dt = cfg.dt

    # 1. Initialize Baseline Estimators
    dirty_x = DirtyDerivativeFilter(tau_d=0.02)
    dirty_th = DirtyDerivativeFilter(tau_d=0.02)

    butter_x = ButterworthDifferentiator()
    butter_th = ButterworthDifferentiator()

    ekf = ContinuousDiscreteEKF(plant)
    rnn = ContinuousShallowRNNObserver(plant)

    # Initial states
    init_meas = meas_hist[0]
    init_state_est = np.array([init_meas[0], 0.0, init_meas[1], 0.0], dtype=np.float64)
    ekf.reset(init_state_est)
    rnn.reset(init_state_est)

    dirty_x.reset(init_meas[0])
    dirty_th.reset(init_meas[1])
    butter_x.reset(init_meas[0])
    butter_th.reset(init_meas[1])

    # Trajectory arrays
    est_dirty = np.zeros((n_steps, 4), dtype=np.float64)
    est_butter = np.zeros((n_steps, 4), dtype=np.float64)
    est_ekf = np.zeros((n_steps, 4), dtype=np.float64)
    est_rnn = np.zeros((n_steps, 4), dtype=np.float64)

    # Custom observer histories if provided
    custom_hist: Dict[str, np.ndarray] = {}
    if custom_observers:
        for name, obs in custom_observers.items():
            custom_hist[name] = np.zeros((n_steps, 4), dtype=np.float64)
            if hasattr(obs, "reset"):
                obs.reset(init_state_est)

    for i in range(n_steps):
        ym = meas_hist[i]
        f_val = force_hist[i]

        # Dirty Derivative: positions directly from measurement, velocities from dirty filter
        v_x_dirty = dirty_x.update(ym[0], dt)
        v_th_dirty = dirty_th.update(ym[1], dt)
        est_dirty[i] = [ym[0], v_x_dirty, ym[1], v_th_dirty]

        # Butterworth Differentiator: 2nd order filter
        v_x_butter = butter_x.update(ym[0], dt)
        v_th_butter = butter_th.update(ym[1], dt)
        est_butter[i] = [ym[0], v_x_butter, ym[1], v_th_butter]

        # EKF
        x_ekf = ekf.step(f_val, ym, dt)
        est_ekf[i] = x_ekf

        # Shallow Dynamic RNN
        x_rnn = rnn.update(ym, f_val, dt)
        est_rnn[i] = x_rnn

        # Custom observers
        if custom_observers:
            for name, obs in custom_observers.items():
                if hasattr(obs, "update"):
                    x_c = obs.update(ym, f_val, dt)
                    custom_hist[name][i] = x_c

    estimates: Dict[str, np.ndarray] = {
        "Dirty_Derivative": est_dirty,
        "Butterworth_Diff": est_butter,
        "Continuous_EKF": est_ekf,
        "Shallow_Dynamic_RNN": est_rnn,
    }
    estimates.update(custom_hist)

    # Evaluate estimation performance
    reports: Dict[str, EstimationReport] = {}
    for name, est_traj in estimates.items():
        reports[name] = evaluate_estimation_performance(t_arr, state_hist, est_traj, dt=dt)

    return SimulationResult(
        t=t_arr,
        state_true=state_hist,
        measurements=meas_hist,
        force_applied=force_hist,
        estimates=estimates,
        reports=reports,
    )
