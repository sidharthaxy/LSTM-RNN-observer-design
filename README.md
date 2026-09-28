# Approach A: Unconstrained Continuous-Time Lb-LSTM Observer

[![Branch](https://img.shields.io/badge/Branch-feature%2Fapproach--a--blackbox--lstm-blue.svg)](#)
[![Paradigms](https://img.shields.io/badge/Paradigm-Black--Box%20Neural%20ODE%20Observer-brightgreen.svg)](#)
[![Tests Passing](https://img.shields.io/badge/tests-21%2F21%20passing-success.svg)](#)

---

## 1. Architectural Overview & Motivation

This branch implements **Approach A**, an **Unconstrained Continuous-Time Lyapunov-Based Long Short-Term Memory (Lb-LSTM) Observer** designed for nonlinear state estimation and dynamic model identification on the **Feedback Instruments 33-936S Digital Cart-Inverted Pendulum Rig**.

In this architecture, **no prior mechanical structure** (such as mass matrices, Coriolis matrices, or gravity vectors) is assumed for the vector field. The continuous-time LSTM network acts as an unconstrained universal approximator of the complete acceleration vector field:

$$\mathbf{a}(\mathbf{x}, u) = \begin{bmatrix} \ddot{x} \\ \ddot{\theta} \end{bmatrix} \in \mathbb{R}^2$$

The observer operates purely in continuous time via coupled ordinary differential equations (ODEs), updating its readout weights dynamically using an **analytical Lyapunov-derived adaptation law with leakage (e-modification)**.

```
       +----------------------- Continuous Lb-LSTM Observer ------------------------+
       |                                                                             |
       |  Feature Input z(t) = [x_hat(t), u(t)]                                      |
       |               |                                                             |
       |               v                                                             |
       |      [ Continuous Gates: i(t), f(t), o(t), g(t) ]                           |
       |               |                                                             |
       |               v                                                             |
       |      Cell State ODE: dc/dt = -(1 - f)*c + i*g                               |
       |               |                                                             |
       |               v                                                             |
       |      Hidden State: h(t) = o(t) * tanh(c(t))                                 |
       |               |                                                             |
       |               v                                                             |
       |      Acceleration: a_hat(t) = W_a(t) * h(t) + b_a                           |
       |               |                                                             |
       |               v                                                             |
       |      State Derivatives: dx_hat/dt = A_nom * x_hat + B_a * a_hat + L * e_y   |
       |                                                                             |
       |      Lyapunov Adaptation: dW_a/dt = Gamma * (e_y * h^T) - sigma_leak * W_a  |
       +-----------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation

### 2.1 Continuous-Time LSTM (CT-LSTM) Formulation

Let the feature vector be $z(t) = [\hat{\mathbf{x}}(t)^T, u(t)]^T \in \mathbb{R}^5$. The continuous-time gating equations are defined as:

$$\begin{aligned}
i(t) &= \sigma\left(\mathbf{W}_i z(t) + \mathbf{b}_i\right) \quad &&\text{(Input Gate)} \\
f(t) &= \sigma\left(\mathbf{W}_f z(t) + \mathbf{b}_f\right) \quad &&\text{(Forget Gate)} \\
o(t) &= \sigma\left(\mathbf{W}_o z(t) + \mathbf{b}_o\right) \quad &&\text{(Output Gate)} \\
g(t) &= \tanh\left(\mathbf{W}_g z(t) + \mathbf{b}_g\right) \quad &&\text{(Candidate Cell State)}
\end{aligned}$$

where $\sigma(\xi) = \frac{1}{1 + e^{-\xi}}$ is the element-wise logistic sigmoid.

The **cell state differential equation** replaces discrete recurrence with a continuous leaky ODE:

$$\frac{d c(t)}{d t} = -\left(\mathbf{1} - f(t)\right) \odot c(t) + i(t) \odot g(t), \quad c(0) = \mathbf{0}$$

The hidden state representation is given by:

$$h(t) = o(t) \odot \tanh(c(t)) \in \mathbb{R}^{d_h}$$

### 2.2 Acceleration Field Approximation

The 2-DOF mechanical acceleration vector is reconstructed via adaptive linear readout:

$$\hat{\mathbf{a}}(t) = \begin{bmatrix} \hat{\ddot{x}}(t) \\ \hat{\ddot{\theta}}(t) \end{bmatrix} = \mathbf{W}_a(t) h(t) + \mathbf{b}_a \in \mathbb{R}^2$$

### 2.3 Observer Kinematic Dynamics with Output Injection

Let $y(t) = [x_m(t), \theta_m(t)]^T \in \mathbb{R}^2$ denote optical encoder measurements. The observer state $\hat{\mathbf{x}} = [\hat{x}, \hat{\dot{x}}, \hat{\theta}, \hat{\dot{\theta}}]^T$ propagates according to:

$$\begin{aligned}
\frac{d \hat{x}_1}{dt} &= \hat{x}_2 + L_1 \left(y_1(t) - \hat{x}_1\right) \\
\frac{d \hat{x}_2}{dt} &= \hat{a}_1(t) + L_2 \left(y_1(t) - \hat{x}_1\right) \\
\frac{d \hat{x}_3}{dt} &= \hat{x}_4 + L_3 \left(y_2(t) - \hat{x}_3\right) \\
\frac{d \hat{x}_4}{dt} &= \hat{a}_2(t) + L_4 \left(y_2(t) - \hat{x}_3\right)
\end{aligned}$$

where $L_1, L_2, L_3, L_4 > 0$ are design Luenberger injection gains placed to guarantee stability of the nominal linear estimation subsystem.

---

## 3. Lyapunov Stability & Adaptation Law

### 3.1 Error System Dynamics

Define the state estimation error $e(t) = \mathbf{x}(t) - \hat{\mathbf{x}}(t)$ and parameter estimation error $\tilde{\mathbf{W}}_a(t) = \mathbf{W}_a^* - \mathbf{W}_a(t)$, where $\mathbf{W}_a^*$ is the optimal unknown parameter matrix satisfying:

$$\mathbf{a}(\mathbf{x}, u) = \mathbf{W}_a^* h(t) + \mathbf{\epsilon}_a(\mathbf{x}, u), \quad \|\mathbf{\epsilon}_a\| \le \epsilon_0$$

The error dynamics become:

$$\dot{e}(t) = \mathbf{A}_L e(t) + \mathbf{B}_a \left( \tilde{\mathbf{W}}_a(t) h(t) + \mathbf{\epsilon}_a \right)$$

where $\mathbf{A}_L$ is strictly Hurwitz:

$$\mathbf{A}_L = \begin{bmatrix}
-L_1 & 1 & 0 & 0 \\
-L_2 & 0 & 0 & 0 \\
0 & 0 & -L_3 & 1 \\
0 & 0 & -L_4 & 0
\end{bmatrix}, \quad
\mathbf{B}_a = \begin{bmatrix}
0 & 0 \\
1 & 0 \\
0 & 0 \\
0 & 1
\end{bmatrix}$$

There exists a unique symmetric positive-definite matrix $\mathbf{P} = \mathbf{P}^T > 0$ solving the continuous Lyapunov equation:

$$\mathbf{A}_L^T \mathbf{P} + \mathbf{P} \mathbf{A}_L = -\mathbf{Q}_L, \quad \mathbf{Q}_L > 0$$

### 3.2 Candidate Lyapunov Function

Choose the positive-definite Lyapunov function candidate:

$$V(e, \tilde{\mathbf{W}}_a) = \frac{1}{2} e^T \mathbf{P} e + \frac{1}{2 \Gamma_a} \text{tr}\left( \tilde{\mathbf{W}}_a^T \tilde{\mathbf{W}}_a \right)$$

Differentiating along trajectories:

$$\dot{V} = -\frac{1}{2} e^T \mathbf{Q}_L e + e^T \mathbf{P} \mathbf{B}_a \mathbf{\epsilon}_a + \text{tr}\left( \tilde{\mathbf{W}}_a^T \left[ \frac{1}{\Gamma_a} \dot{\tilde{\mathbf{W}}}_a + \mathbf{B}_a^T \mathbf{P} e \, h(t)^T \right] \right)$$

### 3.3 Online Weight Adaptation Law

Noticing that $\dot{\tilde{\mathbf{W}}}_a = -\dot{\mathbf{W}}_a$, setting the bracketed expression to cancel and adding a continuous $\sigma$-leakage (e-modification) term yields:

$$\dot{\mathbf{W}}_a(t) = \Gamma_a \left[ \mathbf{e}_y(t) h(t)^T \right] - \sigma_{\text{leak}} \mathbf{W}_a(t)$$

where $\mathbf{e}_y(t) = [y_1 - \hat{x}_1, y_2 - \hat{x}_3]^T = \mathbf{B}_a^T \mathbf{P} e(t)$ with diagonal $\mathbf{P}$.

**Theorem (Uniform Ultimate Boundedness)**: Under the continuous adaptation law above, all closed-loop signals $e(t)$ and $\mathbf{W}_a(t)$ remain strictly bounded for all $t \ge 0$, and the estimation error norm converges exponentially to a compact residual set $\mathcal{D}_e = \{e \in \mathbb{R}^4 : \|e\| \le \mu\}$.

---

## 4. Source Files on this Branch

- `src/observers/blackbox_lstm_observer.py`: Complete continuous-time implementation of `BlackBoxLSTMObserver` and `BlackBoxLSTMConfig`.
- `tests/test_blackbox_lstm.py`: Unit tests validating state reset, stability, and weight boundedness.
- `src/plant/`: Feedback 33-936S ground truth simulation and PCI-1711 DAQ noise injection.
- `src/baselines/`: Classical baseline estimators for comparative evaluation.

---

## 5. Quickstart & Verification

```bash
# Verify unit tests for Approach A
pytest tests/test_blackbox_lstm.py -v

# Run full test suite
pytest -v
```
