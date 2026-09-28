# Approach C: Concurrent Learning Lb-LSTM Adaptive Observer

[![Branch](https://img.shields.io/badge/Branch-feature%2Fapproach--c--concurrent--learning-red.svg)](#)
[![Paradigms](https://img.shields.io/badge/Paradigm-Concurrent%20Learning%20SciML-brightgreen.svg)](#)
[![Tests Passing](https://img.shields.io/badge/tests-21%2F21%20passing-success.svg)](#)

---

## 1. Architectural Overview & Motivation

This branch implements **Approach C**, a **Concurrent Learning Lyapunov-Based LSTM (CL-LSTM) Adaptive Observer** for the **Feedback Instruments 33-936S Digital Cart-Inverted Pendulum Rig**.

### The Fundamental PE Bottleneck in Adaptive Observers
In standard adaptive control and neural observers (including Approaches A and B), parameter error convergence $\tilde{\mathbf{W}}(t) \to \mathbf{0}$ mathematically requires the reference trajectory to satisfy the **Persistence of Excitation (PE)** condition:

$$\int_{t}^{t+T_0} \phi(\tau)\phi(\tau)^T d\tau \ge \alpha_0 \mathbf{I} > 0 \quad \forall t \ge 0$$

In cart-inverted pendulum stabilization, once the cart reaches steady state and the pendulum stabilizes upright, velocities and tracking errors decay to zero ($\dot{x} \to 0, \theta \to 0, u \to 0$). As a result, the regressor vector $\phi(t)$ collapses:

$$\lim_{t \to \infty} \phi(t) = \mathbf{0} \implies \text{PE Condition is Completely Violated}$$

Under classical adaptation laws, weights stop adapting, freeze, or suffer from parameter drift and unlearning under sensor noise.

### The Concurrent Learning Solution
**Concurrent Learning (CL)** (Chowdhary et al., 2013) solves this fundamental limitation by augmenting the continuous-time observer with an online **History Stack** $\mathcal{H} = \{(h_j, \mathbf{a}_j)\}_{j=1}^{N_H}$ of rich past state transitions. By simultaneously adapting on both **instantaneous tracking error** and **stored historical residuals**, the observer achieves **exponential parameter convergence $\tilde{\mathbf{W}} \to \mathbf{0}$ without requiring ongoing persistent excitation**.

```
       +------------------ Concurrent Learning Lb-LSTM Observer -------------------+
       |                                                                           |
       |  Continuous LSTM: z(t) = [x_hat(t), u(t)]                                 |
       |               |                                                           |
       |               v                                                           |
       |      Cell State ODE: dc/dt = -(1 - f)*c + i*g                             |
       |      Hidden State:   h(t) = o(t) * tanh(c(t))                             |
       |               |                                                           |
       |               +---> Instantaneous Innovation: e_y(t) = y(t) - y_hat(t)    |
       |               |                                                           |
       |               +---> Online History Stack H = {(h_j, a_j)}_{j=1}^{N_H}     |
       |                     Selection: lambda_min( sum h_j * h_j^T ) >= lambda > 0|
       |                     Residual: R_CL = sum_{j=1}^{N_H} (a_j - W*h_j) * h_j^T|
       |                                                                           |
       |  Composite Lyapunov Adaptation Law:                                       |
       |  dW_a/dt = Gamma_inst * [ e_y(t) * h(t)^T ]                               |
       |          + Gamma_cl   * sum_{j=1}^{N_H} [ (a_j - W_a * h_j) * h_j^T ]      |
       |          - sigma_leak * W_a                                               |
       |                                                                           |
       |  Result: Exponential Parameter Convergence Even at Steady Equilibrium!    |
       +---------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation & Rank Maximization

### 2.1 Online History Stack Conditioning

Let $\mathcal{H} = \{(h_j, \mathbf{a}_j)\}_{j=1}^{N_H}$ represent the set of stored transitions, where $h_j = h(t_j) \in \mathbb{R}^{d_h}$ is the recorded hidden state representation and $\mathbf{a}_j = \mathbf{a}(t_j) \in \mathbb{R}^2$ is the target acceleration.

Define the history stack excitation matrix:

$$\mathbf{\Omega} = \sum_{j=1}^{N_H} h_j h_j^T \in \mathbb{R}^{d_h \times d_h}$$

A candidate transition $(h_{\text{new}}, \mathbf{a}_{\text{new}})$ is admitted into the history stack if and only if it enhances or preserves the **minimum singular value / eigenvalue**:

$$\lambda_{\min}(\mathbf{\Omega}) \ge \underline{\lambda} > 0$$

When the stack reaches maximum capacity $N_H$, a cyclic replacement algorithm evaluates candidate substitutions to strictly maximize $\det(\mathbf{\Omega})$ and $\lambda_{\min}(\mathbf{\Omega})$.

---

## 3. Composite Lyapunov Stability & Parameter Convergence Proof

### 3.1 Composite Error Dynamics

Let $e(t) = \mathbf{x}(t) - \hat{\mathbf{x}}(t)$ be the state error and $\tilde{\mathbf{W}}_a = \mathbf{W}_a^* - \mathbf{W}_a$ be the parameter estimation error.

On the history stack points, the residual is:

$$\mathbf{a}_j - \hat{\mathbf{a}}_j = \mathbf{W}_a^* h_j + \mathbf{\epsilon}_j - (\mathbf{W}_a h_j + \mathbf{b}_a) = \tilde{\mathbf{W}}_a h_j + \mathbf{\epsilon}_j$$

Multiplying by $h_j^T$:

$$\sum_{j=1}^{N_H} (\mathbf{a}_j - \hat{\mathbf{a}}_j) h_j^T = \tilde{\mathbf{W}}_a \left( \sum_{j=1}^{N_H} h_j h_j^T \right) + \sum_{j=1}^{N_H} \mathbf{\epsilon}_j h_j^T = \tilde{\mathbf{W}}_a \mathbf{\Omega} + \mathbf{E}_{\mathcal{H}}$$

### 3.2 Candidate Lyapunov Function

Choose the composite Lyapunov candidate:

$$V(e, \tilde{\mathbf{W}}_a) = \frac{1}{2} e^T \mathbf{P} e + \frac{1}{2 \Gamma_{\text{inst}}} \text{tr}\left( \tilde{\mathbf{W}}_a^T \tilde{\mathbf{W}}_a \right)$$

### 3.3 Negative Definiteness via Full-Rank History Stack

Evaluating the time derivative $\dot{V}$ along the closed-loop system:

$$\dot{V} = -\frac{1}{2} e^T \mathbf{Q} e + e^T \mathbf{P} \mathbf{B}_a \mathbf{\epsilon}_0 + \text{tr}\left( \tilde{\mathbf{W}}_a^T \left[ \frac{1}{\Gamma_{\text{inst}}} \dot{\tilde{\mathbf{W}}}_a + \mathbf{e}_y h(t)^T \right] \right)$$

Substituting the composite Concurrent Learning adaptation law:

$$\dot{\mathbf{W}}_a(t) = \Gamma_{\text{inst}} \left[ \mathbf{e}_y(t) h(t)^T \right] + \Gamma_{\text{cl}} \sum_{j=1}^{N_H} (\mathbf{a}_j - \mathbf{W}_a h_j) h_j^T - \sigma_{\text{leak}} \mathbf{W}_a$$

Since $\dot{\tilde{\mathbf{W}}}_a = -\dot{\mathbf{W}}_a$:

$$\dot{V} = -\frac{1}{2} e^T \mathbf{Q} e - \frac{\Gamma_{\text{cl}}}{\Gamma_{\text{inst}}} \text{tr}\left( \tilde{\mathbf{W}}_a^T \tilde{\mathbf{W}}_a \mathbf{\Omega} \right) + \text{perturbation terms}$$

Using the trace inequality $\text{tr}(\tilde{\mathbf{W}}_a^T \tilde{\mathbf{W}}_a \mathbf{\Omega}) \ge \lambda_{\min}(\mathbf{\Omega}) \|\tilde{\mathbf{W}}_a\|_F^2$:

$$\dot{V} \le -\frac{1}{2} \lambda_{\min}(\mathbf{Q}) \|e\|^2 - \frac{\Gamma_{\text{cl}} \underline{\lambda}}{\Gamma_{\text{inst}}} \|\tilde{\mathbf{W}}_a\|_F^2 + \delta_0$$

**Theorem (Exponential Convergence without PE)**: If the history stack satisfies $\lambda_{\min}(\mathbf{\Omega}) \ge \underline{\lambda} > 0$, the state and parameter errors $(e(t), \tilde{\mathbf{W}}_a(t))$ converge exponentially to a compact residual ball proportional to modelling inaccuracies. When the reference trajectory stops moving ($\phi(t) \to \mathbf{0}$), the history stack term maintains strict negative definiteness, preventing parameter drift and guaranteeing parameter retention.

---

## 4. Source Files on this Branch

- `src/observers/concurrent_learning_lstm_observer.py`: Complete implementation of `ConcurrentLearningLSTMObserver`, `HistoryStackEntry`, and rank-conditioning algorithms.
- `tests/test_concurrent_learning_lstm.py`: Unit tests validating history stack acquisition, non-zero excitation matrix, and parameter stability under fading excitation.

---

## 5. Verification Commands

```bash
# Test history stack conditioning & fading excitation stability
pytest tests/test_concurrent_learning_lstm.py -v

# Run full test suite
pytest -v
```
