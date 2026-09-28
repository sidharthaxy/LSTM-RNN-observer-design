# Lyapunov-Based LSTM Adaptive Observers for Underactuated Cart-Inverted Pendulum Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scientific Stack](https://img.shields.io/badge/SciML-NumPy%20%7C%20SciPy%20%7C%20Numba-orange.svg)](#)
[![Rig: Feedback 33-936S](https://img.shields.io/badge/Benchmarked-Feedback%2033--936S-green.svg)](#)
[![Tests Passing](https://img.shields.io/badge/tests-19%2F19%20passing-brightgreen.svg)](#)

---

## 1. Executive Summary & Research Vision

This repository hosts a scientific control and machine learning research platform investigating **Lyapunov-Based Long Short-Term Memory (Lb-LSTM) Continuous Adaptive Observers** for nonlinear state reconstruction and real-time dynamic model identification. 

The experimental benchmark system is the industry-standard **Feedback Instruments 33-936S Digital Cart-Inverted Pendulum Rig**, interfaced via an **Advantech PCI-1711 DAQ card**. In real-world operation, control engineers only have access to discrete, quantized, and noisy optical encoder positions ($x, \theta$). Direct analytical differentiation (e.g., dirty derivatives or high-gain observers) causes extreme noise amplification and high-frequency chattering, degrading closed-loop stabilization and actuator lifespan.

This project formulates **continuous-time neural adaptive observers** whose weight adaptation laws are analytically derived using **Lyapunov stability theory**. Unlike black-box deep learning models that execute discrete unconstrained backpropagation through time, our observers run **continuous analytical ordinary differential equations (ODEs)** evaluated in real time via high-performance NumPy/Numba routines with strict mathematical guarantees of uniform ultimate boundedness (UUB) or asymptotic convergence.

---

## 2. Theoretical Problem Statement

Consider the generalized underactuated Euler-Lagrange mechanical system:

$$\mathbf{M}(q)\ddot{q} + \mathbf{C}(q, \dot{q})\dot{q} + \mathbf{G}(q) + \mathbf{F}_f(\dot{q}) + \mathbf{\Delta}(q, \dot{q}, t) = \mathbf{B} u$$

with generalized coordinates $q = [x, \theta]^T \in \mathbb{R}^2$, actuator control force $u = F \in \mathbb{R}$, and state vector $\mathbf{x} = [x, \dot{x}, \theta, \dot{\theta}]^T = [x_1, x_2, x_3, x_4]^T \in \mathbb{R}^4$.

### Sensor Constraints and Unmeasured Velocity
Only configuration positions are measured through digital optical encoders:

$$y(t) = \begin{bmatrix} x_m(t) \\ \theta_m(t) \end{bmatrix} = \mathbf{H} \mathbf{x}(t) + \mathbf{v}(t) = \begin{bmatrix} x(t) \\ \theta(t) \end{bmatrix} + \begin{bmatrix} v_x(t) \\ v_\theta(t) \end{bmatrix}$$

where:
1. **Optical Encoder Noise**: $v_\theta \sim \mathcal{U}(-0.5^\circ, +0.5^\circ)$ and $v_x \sim \mathcal{U}(-0.2\text{ mm}, +0.2\text{ mm})$.
2. **Quantization**: Quadrature counters on the PCI-1711 DAQ discretize position ($4096\text{ counts/m}$ for cart, $4096\text{ counts/rev}$ for pendulum).
3. **Asymmetric Dead-Zone & Stiction**: Drive amplifier exhibits an asymmetric voltage deadband ($+0.15\text{ V}$ forward, $-0.12\text{ V}$ reverse) and static breakaway friction ($F_{\text{stiction}} \approx 0.35\text{ N}$).
4. **Unmodeled Nonlinear Dynamics**: $\mathbf{\Delta}(q, \dot{q}, t)$ captures cable drag, track irregularities, and velocity-dependent stiction.

**Objective**: Reconstruct full state $\hat{\mathbf{x}} = [\hat{x}, \hat{\dot{x}}, \hat{\theta}, \hat{\dot{\theta}}]^T$ and identify unknown vector fields while eliminating velocity chattering and guaranteeing bounded estimation error $\lim_{t \to \infty} \|\mathbf{x}(t) - \hat{\mathbf{x}}(t)\| \le \epsilon$.

---

## 3. Mathematical Ground Truth: Feedback Instruments 33-936S Rig

The equations of motion are sourced directly from the **Feedback Instruments 33-936S manufacturer manual**:

$$\begin{aligned}
(M + m)\ddot{x} + b\dot{x} + m l \ddot{\theta}\cos(\theta) - m l \dot{\theta}^2\sin(\theta) &= F \\
(I + m l^2)\ddot{\theta} - m g l \sin(\theta) + m l \ddot{x}\cos(\theta) + d\dot{\theta} &= 0
\end{aligned}$$

### Matrix Formulation

$$\begin{bmatrix} M + m & m l \cos\theta \\ m l \cos\theta & I + m l^2 \end{bmatrix} \begin{bmatrix} \ddot{x} \\ \ddot{\theta} \end{bmatrix} = \begin{bmatrix} F - b\dot{x} + m l \dot{\theta}^2\sin\theta \\ m g l \sin\theta - d\dot{\theta} \end{bmatrix}$$

$$\mathbf{M}(\theta) = \begin{bmatrix} M + m & m l \cos\theta \\ m l \cos\theta & I + m l^2 \end{bmatrix}$$

$$\det(\mathbf{M}(\theta)) = (M + m)(I + m l^2) - (m l \cos\theta)^2 > 0 \quad \forall \theta \in \mathbb{R}$$

Because $\det(\mathbf{M}(\theta)) \ge (M+m)(I+ml^2) - m^2 l^2 > 0$, the mass matrix is **strictly symmetric positive-definite** everywhere in state space.

### Analytic Forward Accelerations

$$\begin{aligned}
\ddot{x} &= \frac{(I + m l^2)\left[F - b\dot{x} + m l \dot{\theta}^2\sin\theta\right] - (m l \cos\theta)\left[m g l \sin\theta - d\dot{\theta}\right]}{\det(\mathbf{M}(\theta))} \\
\ddot{\theta} &= \frac{-(m l \cos\theta)\left[F - b\dot{x} + m l \dot{\theta}^2\sin\theta\right] + (M + m)\left[m g l \sin\theta - d\dot{\theta}\right]}{\det(\mathbf{M}(\theta))}
\end{aligned}$$

### Verified Plant Parameter Specifications

| Parameter | Symbol | Nominal Value | Units | Description |
|:---|:---:|:---:|:---:|:---|
| Cart Mass | $M$ | `2.4` | $\text{kg}$ | Linear carriage mass |
| Pendulum Mass | $m$ | `0.23` | $\text{kg}$ | Cylindrical rod mass |
| Center of Mass | $l$ | `0.36` | $\text{m}$ | Pivot to center of mass distance |
| Pendulum Inertia | $I$ | `0.099` | $\text{kg}\cdot\text{m}^2$ | Moment of inertia about pivot |
| Cart Damping | $b$ | `0.05` | $\text{N}\cdot\text{s/m}$ | Track viscous damping |
| Joint Damping | $d$ | `0.005` | $\text{N}\cdot\text{m}\cdot\text{s/rad}$ | Pivot bearing viscous friction |
| Gravity | $g$ | `9.81` | $\text{m/s}^2$ | Gravitational acceleration |
| Max Force | $F_{\text{max}}$ | `20.0` | $\text{N}$ | Drive amplifier force saturation ($\pm 10\text{V}$) |
| Track Limits | $x_{\text{lim}}$ | `0.5` | $\text{m}$ | Hard end-stop bumper travel limit ($\pm 0.5\text{m}$) |

### Exact Analytical Jacobians $\mathbf{A}(\mathbf{x}, u)$ and $\mathbf{B}(\mathbf{x})$

For continuous Extended Kalman Filtering and linearized control, the continuous-time system Jacobian $\mathbf{A} = \frac{\partial f}{\partial \mathbf{x}} \in \mathbb{R}^{4 \times 4}$ and input vector $\mathbf{B} = \frac{\partial f}{\partial u} \in \mathbb{R}^{4 \times 1}$ are derived in closed form:

$$\mathbf{A} = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & \frac{\partial \ddot{x}}{\partial \dot{x}} & \frac{\partial \ddot{x}}{\partial \theta} & \frac{\partial \ddot{x}}{\partial \dot{\theta}} \\
0 & 0 & 0 & 1 \\
0 & \frac{\partial \ddot{\theta}}{\partial \dot{x}} & \frac{\partial \ddot{\theta}}{\partial \theta} & \frac{\partial \ddot{\theta}}{\partial \dot{\theta}}
\end{bmatrix}, \quad
\mathbf{B} = \begin{bmatrix}
0 \\
\frac{I + m l^2}{\det(\mathbf{M})} \\
0 \\
\frac{-m l \cos\theta}{\det(\mathbf{M})}
\end{bmatrix}$$

All partial derivatives are unit-tested against central finite differences to $10^{-5}$ precision in `tests/test_plant.py`.

---

## 4. Architectural Comparison Matrix

| Estimator Paradigm | Formulation Type | Stability Guarantee | Prior Model Required | PE Condition Needed? | Noise & Chattering Robustness | Real-Time Latency | Primary Weakness |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Dirty Derivative Filter** | 1st-order linear filter $\frac{s}{\tau_d s + 1}$ | Asymptotically stable filter poles | None | No | **Extremely Poor** (Chattering Ratio $> 18\times$) | $< 1\ \mu\text{s}$ | Massive high-frequency noise amplification; unmitigated phase lag |
| **Butterworth Differentiator** | 2nd-order bandpass filter $\frac{10^4 s}{s^2 + 70.7s + 10^4}$ | Hurwitz transfer function | None | No | **Moderate** (Chattering Ratio $\approx 3.0\times$) | $< 2\ \mu\text{s}$ | Fixed bandwidth trades phase delay against high-frequency cutoff |
| **Continuous-Discrete EKF** | Non-linear Riccati ODE + discrete update | Local exponential stability under small noise | Exact physical parameters $(M, m, l, I, b, d)$ | No | **Good** (Chattering Ratio $\approx 2.9\times$) | $\approx 25\ \mu\text{s}$ | Sensitive to unmodeled stiction & parameter drift |
| **Shallow Dynamic RNN** (Dinh et al., 2014) | Continuous RNN + Lyapunov leakage | Semi-global Uniform Ultimate Boundedness (UUB) | Nominal plant structure | No (bounded weights) | **High** (Chattering Ratio $\approx 1.0\times$) | $\approx 15\ \mu\text{s}$ | Limited representation capacity for long-term state dependencies |
| **Approach A: Black-Box Lb-LSTM** (`feature/approach-a...`) | Continuous-time LSTM ODE + Lyapunov update | Uniform Ultimate Boundedness (UUB) via Krasovskii/Lyapunov | Black-box acceleration model | No | **Superior** (internal gating memory smooths transients) | $\approx 45\ \mu\text{s}$ | Unconstrained acceleration space can produce transient physical violations |
| **Approach B: Physics-Informed PI-LSTM** (`feature/approach-b...`) | Structural Cholesky inertia $\mathbf{L}\mathbf{L}^T + \mathbf{C}$ skew-symmetry | Strict Lyapunov asymptotic stability via energy-passivity | Structural Euler-Lagrange properties | No | **Maximum** (physically constrained state space) | $\approx 60\ \mu\text{s}$ | Requires structural coordinate projection |
| **Approach C: Concurrent Learning Lb-LSTM** (`feature/approach-c...`) | Lb-LSTM + Rank-conditioned History Stack | Exponential Parameter Convergence $\tilde{W} \to 0$ | History stack of rich state-action pairs | **Relaxed** (No persistent excitation of trajectory required) | **Maximum** (fast convergence without continuous excitation) | $\approx 80\ \mu\text{s}$ | History stack management overhead and rank verification |

---

## 5. Repository Architecture & Isolated Feature Branches

The project maintains a clean **isolated branch architecture** corresponding to the three parallel scientific paradigms:

```
├── main (Shared Testbed, Plant Truth, Noise Model, Baselines, Metrics)
│
├── feature/approach-a-blackbox-lstm
│   └── Continuous-time Lb-LSTM observer approximating the complete acceleration vector field.
│
├── feature/approach-b-physics-informed
│   └── PI-LSTM enforcing positive-definite inertia L(q)L(q)^T and Coriolis skew-symmetry.
│
└── feature/approach-c-concurrent-learning
    └── Lb-LSTM augmented with an online history stack for parameter convergence without PE.
```

### Main Branch Directory Structure

```
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/
│   ├── plant/
│   │   ├── pendulum_plant.py       # Feedback 33-936S non-linear ODE plant & Jacobians
│   │   └── sensor_noise.py         # PCI-1711 DAQ noise, stiction, and quantization
│   ├── baselines/
│   │   └── classical_estimators.py # Dirty Derivative, Butterworth, EKF, Shallow RNN
│   ├── utils/
│   │   └── metrics.py              # RMSE, steady-state norm, peak overshoot, THD, Chattering
│   └── simulation/
│       └── benchmark_runner.py     # LQR closed-loop testbed & signal excitation runner
├── tests/
│   ├── test_plant.py               # Dynamics, Jacobians (analytical vs numerical), energy
│   ├── test_sensor_noise.py        # Noise bounds, quantization, stiction deadband
│   ├── test_baselines.py           # Estimator tracking, convergence, and boundedness
│   └── test_metrics.py             # Validation of RMSE, THD, and Chattering metrics
├── scripts/
│   ├── run_benchmarks.py           # Command-line benchmark execution
│   └── visualize_baselines.py      # Publication-quality figure generation
└── figures/
    └── baseline_estimation_comparison.png
```

---

## 6. Quickstart Guide

### 6.1 Environment Setup

Create and activate a virtual environment with Python $\ge 3.10$:

```bash
# Clone the repository
git clone https://github.com/sidharthaxy/LSTM-RNN-observer-design.git
cd LSTM-RNN-observer-design

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install scientific dependencies and test suite
pip install -r requirements.txt
```

> **Note on Machine Learning Dependencies**: Core observer updates do **not** depend on PyTorch or TensorFlow. All continuous Lyapunov update laws are analytical differential equations evaluated in real time via high-speed NumPy and SciPy routines.

### 6.2 Running the Test Suite

Run the full pytest suite (19 unit tests validating plant dynamics, numerical Jacobians, sensor noise, estimators, and metrics):

```bash
pytest -v
```

### 6.3 Running Benchmarks

Execute the automated benchmark across all four baseline estimators under PCI-1711 DAQ noise and asymmetric stiction:

```bash
python scripts/run_benchmarks.py
```

Sample Benchmark Output (5000 time steps @ 1 kHz):

```text
================================================================================
FEEDBACK INSTRUMENTS 33-936S CART-INVERTED PENDULUM BENCHMARK TESTBED
Running Baseline Estimator Comparisons under PCI-1711 DAQ Noise & Stiction...
================================================================================

[+] Benchmark Completed across 4 Estimators (5000 time steps @ 1 kHz):
    - True plant states: x, x_dot, theta, theta_dot
    - Measurements: Optical Encoders with uniform noise & quantization
    - Non-linear effects: Asymmetric stiction deadband [+0.15V, -0.12V]

                     Cart_Pos_RMSE [m]  Cart_Vel_RMSE [m/s]  Theta_RMSE [rad]  Theta_Dot_RMSE [rad/s]  Vel_Chattering_Ratio  AngVel_Chattering_Ratio
Estimator                                                                                                                                           
Dirty_Derivative              0.000134             0.014140          0.005065                0.243687             18.964493              3545.064559
Butterworth_Diff              0.000134             0.004776          0.005065                0.135140              3.026037               547.372044
Continuous_EKF                0.000043             0.002251          0.001531                0.073982              2.911799               377.432751
Shallow_Dynamic_RNN           0.000149             0.000322          0.000782                0.003743              0.999903                10.893497

================================================================================
```

### 6.4 Generating Benchmark Plots

Generate publication-quality figures comparing ground truth trajectories against estimator outputs:

```bash
python scripts/visualize_baselines.py
```

Output saved to `figures/baseline_estimation_comparison.png`.

---

## 7. Navigating Research Branches

To inspect and run each of the three specialized Lb-LSTM research architectures, switch to the dedicated feature branches:

```bash
# Approach A: Unconstrained Continuous Black-Box Lb-LSTM
git checkout feature/approach-a-blackbox-lstm

# Approach B: Physics-Informed PI-LSTM (Positive-Definite Inertia & Skew-Symmetry)
git checkout feature/approach-b-physics-informed

# Approach C: Concurrent Learning Lb-LSTM (History Stack without Persistent Excitation)
git checkout feature/approach-c-concurrent-learning
```

Each branch maintains its own dedicated implementation, mathematical Lyapunov stability proofs, and standalone `README.md`.

---

## References

1. Feedback Instruments Ltd. *Digital Pendulum Control Experiments (Feedback 33-936S)*, Crowborough, UK.
2. Dinh, T. Q., et al. (2014). *Dynamic neural network observers for nonlinear systems with unknown dynamics*, IEEE Transactions on Neural Networks and Learning Systems.
3. Chowdhary, G., et al. (2013). *Concurrent learning adaptive control of linear systems with unknown dynamics*, International Journal of Adaptive Control and Signal Processing.
4. Khalil, H. K. (2002). *Nonlinear Systems* (3rd ed.), Prentice Hall.
