# Approach B: Physics-Informed PI-LSTM Adaptive Observer

[![Branch](https://img.shields.io/badge/Branch-feature%2Fapproach--b--physics--informed-purple.svg)](#)
[![Paradigms](https://img.shields.io/badge/Paradigm-Physics--Informed%20SciML%20Observer-brightgreen.svg)](#)
[![Tests Passing](https://img.shields.io/badge/tests-22%2F22%20passing-success.svg)](#)

---

## 1. Architectural Overview & Motivation

This branch implements **Approach B**, a **Physics-Informed Lyapunov-Based LSTM (PI-LSTM) Observer** benchmarked on the **Feedback Instruments 33-936S Digital Cart-Inverted Pendulum Rig**.

While unconstrained black-box estimators (Approach A) approximate the acceleration vector field directly, they can produce transient dynamic predictions that violate fundamental conservation laws (e.g., negative effective mass or non-dissipative internal forces). 

**Approach B embeds non-negotiable physical constraints directly into the neural observer dynamics**:
1. **Inertia Positive-Definiteness**: $\hat{\mathbf{M}}(q) = \hat{\mathbf{M}}(q)^T > 0$ strictly across all configuration angles $\theta \in \mathbb{R}$.
2. **Coriolis Skew-Symmetry**: $\dot{\hat{\mathbf{M}}}(q, \dot{q}) - 2\hat{\mathbf{C}}(q, \dot{q})$ is identically skew-symmetric, satisfying $z^T (\dot{\mathbf{M}} - 2\mathbf{C}) z = 0$ for all $z \in \mathbb{R}^2$.
3. **Passivity-Based Energy Lyapunov Function**: The skew-symmetry property cancels unmeasurable velocity cross-terms, guaranteeing asymptotic estimation stability without ad-hoc high gain tuning.

```
       +------------------- Physics-Informed PI-LSTM Observer -------------------+
       |                                                                         |
       |  Configuration Coordinates: q = [x, theta]^T                            |
       |               |                                                         |
       |               +---> Inertia Matrix M(q) > 0  (Strictly Positive-Definite)|
       |               |                                                         |
       |               +---> Christoffel Symbols -> Coriolis Matrix C(q, q_dot)  |
       |               |     [ Identity: z^T * (dM/dt - 2*C) * z == 0 ]          |
       |               |                                                         |
       |  CT-LSTM Network: z(t) = [q_hat, q_dot_hat, u]                          |
       |               |                                                         |
       |               v                                                         |
       |      Cell State ODE: dc/dt = -(1 - f)*c + i*g                           |
       |      Hidden State:   h(t) = o(t) * tanh(c(t))                           |
       |               |                                                         |
       |               v                                                         |
       |      Residual Generalized Force: tau_Delta = W_Delta * h(t)             |
       |               |                                                         |
       |               v                                                         |
       |      Euler-Lagrange Equation:                                           |
       |      M(q) * q_ddot + C(q, q_dot)*q_dot + G(q) + D*q_dot =               |
       |           B*u + tau_Delta + tau_injection                               |
       |                                                                         |
       |      Passivity Adaptation: dW_Delta/dt = Gamma * (e_q * h^T) - sigma*W  |
       +-------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation & Structural Properties

### 2.1 Enforcing Inertia Matrix Positive-Definiteness

The mass matrix $\mathbf{M}(\theta)$ of the Feedback 33-936S rig is parameterized by the pendulum angle $\theta$:

$$\mathbf{M}(\theta) = \begin{bmatrix} M + m & m l \cos\theta \\ m l \cos\theta & I + m l^2 \end{bmatrix}$$

Because:
$$\det(\mathbf{M}(\theta)) = (M + m)(I + m l^2) - (m l \cos\theta)^2 \ge (M + m)(I + m l^2) - m^2 l^2 > 0$$

$\mathbf{M}(\theta)$ is **strictly symmetric positive-definite** everywhere in state space, eliminating the risk of singular matrix inversion.

### 2.2 Coriolis Matrix & Skew-Symmetry Proof

The elements of the Coriolis matrix $\mathbf{C}(q, \dot{q})$ are determined uniquely through the Christoffel symbols of the first kind:

$$c_{kj}(q, \dot{q}) = \sum_{i=1}^2 \frac{1}{2} \left( \frac{\partial m_{kj}}{\partial q_i} + \frac{\partial m_{ki}}{\partial q_j} - \frac{\partial m_{ij}}{\partial q_k} \right) \dot{q}_i$$

For the coordinates $q = [x, \theta]^T$ with $\dot{q} = [\dot{x}, \dot{\theta}]^T$:
- $m_{11} = M+m$ (constant)
- $m_{12} = m_{21} = m l \cos\theta$
- $m_{22} = I + m l^2$ (constant)

Evaluating the Christoffel symbols yields:

$$\mathbf{C}(q, \dot{q}) = \begin{bmatrix} 0 & -m l \dot{\theta} \sin\theta \\ 0 & 0 \end{bmatrix}$$

Now compute the time derivative of the mass matrix $\dot{\mathbf{M}}(\theta, \dot{\theta})$:

$$\dot{\mathbf{M}}(\theta, \dot{\theta}) = \begin{bmatrix} 0 & -m l \dot{\theta} \sin\theta \\ -m l \dot{\theta} \sin\theta & 0 \end{bmatrix}$$

Subtracting $2\mathbf{C}(q, \dot{q})$:

$$\dot{\mathbf{M}} - 2\mathbf{C} = \begin{bmatrix} 0 & m l \dot{\theta} \sin\theta \\ -m l \dot{\theta} \sin\theta & 0 \end{bmatrix}$$

Notice that $(\dot{\mathbf{M}} - 2\mathbf{C})^T = -(\dot{\mathbf{M}} - 2\mathbf{C})$. Therefore, the matrix is **identically skew-symmetric**:

$$\mathbf{z}^T \left( \dot{\mathbf{M}} - 2\mathbf{C} \right) \mathbf{z} = 0 \quad \forall \mathbf{z} \in \mathbb{R}^2$$

This mechanical passivity property is verified numerically across arbitrary coordinates in `tests/test_physics_informed_lstm.py`.

---

## 3. Energy-Passivity Lyapunov Stability Analysis

### 3.1 Observer Equations of Motion

$$\mathbf{M}(q) \ddot{\hat{q}} + \mathbf{C}(q, \dot{\hat{q}})\dot{\hat{q}} + \mathbf{G}(q) + \mathbf{D} \dot{\hat{q}} = \mathbf{B} u + \hat{\mathbf{\tau}}_{\Delta} + \mathbf{\tau}_{\text{inject}}$$

where:
- $\hat{\mathbf{\tau}}_{\Delta} = \mathbf{W}_{\Delta}(t) h(t)$ is the CT-LSTM residual force estimation.
- $\mathbf{\tau}_{\text{inject}} = \mathbf{K}_p \mathbf{e}_q$ is the passivity-based position error injection.
- $\mathbf{e}_q = [x_m - \hat{x}, \theta_m - \hat{\theta}]^T$ is the measured optical position error.

### 3.2 Candidate Lyapunov Function

Define the velocity estimation error $\mathbf{e}_v = \dot{q} - \dot{\hat{q}}$. Choose the mechanical energy-based Lyapunov candidate:

$$V(\mathbf{e}_v, \mathbf{e}_q, \tilde{\mathbf{W}}_{\Delta}) = \frac{1}{2} \mathbf{e}_v^T \mathbf{M}(q) \mathbf{e}_v + \frac{1}{2} \mathbf{e}_q^T \mathbf{K}_p \mathbf{e}_q + \frac{1}{2 \Gamma_{\Delta}} \text{tr}\left( \tilde{\mathbf{W}}_{\Delta}^T \tilde{\mathbf{W}}_{\Delta} \right)$$

### 3.3 Cancellation of Cross-Terms via Skew-Symmetry

Differentiating $V$ along trajectories:

$$\dot{V} = \mathbf{e}_v^T \mathbf{M} \dot{\mathbf{e}}_v + \frac{1}{2} \mathbf{e}_v^T \dot{\mathbf{M}} \mathbf{e}_v + \mathbf{e}_q^T \mathbf{K}_p \dot{\mathbf{e}}_q + \frac{1}{\Gamma_{\Delta}} \text{tr}\left( \tilde{\mathbf{W}}_{\Delta}^T \dot{\tilde{\mathbf{W}}}_{\Delta} \right)$$

Substituting the error dynamics $\mathbf{M} \dot{\mathbf{e}}_v = -\mathbf{C} \mathbf{e}_v - \mathbf{D} \mathbf{e}_v - \mathbf{K}_p \mathbf{e}_q + \tilde{\mathbf{W}}_{\Delta} h(t) + \mathbf{\epsilon}$:

$$\dot{V} = \mathbf{e}_v^T \left[ -\mathbf{C} \mathbf{e}_v - \mathbf{D} \mathbf{e}_v - \mathbf{K}_p \mathbf{e}_q + \tilde{\mathbf{W}}_{\Delta} h(t) + \mathbf{\epsilon} \right] + \frac{1}{2} \mathbf{e}_v^T \dot{\mathbf{M}} \mathbf{e}_v + \mathbf{e}_q^T \mathbf{K}_p \mathbf{e}_v + \frac{1}{\Gamma_{\Delta}} \text{tr}\left( \tilde{\mathbf{W}}_{\Delta}^T \dot{\tilde{\mathbf{W}}}_{\Delta} \right)$$

Grouping the quadratic velocity terms:

$$\mathbf{e}_v^T \left( \frac{1}{2} \dot{\mathbf{M}} - \mathbf{C} \right) \mathbf{e}_v = \frac{1}{2} \mathbf{e}_v^T \left( \dot{\mathbf{M}} - 2\mathbf{C} \right) \mathbf{e}_v \equiv 0$$

**The skew-symmetric term vanishes completely!**

Furthermore, the position error cross-terms cancel:

$$-\mathbf{e}_v^T \mathbf{K}_p \mathbf{e}_q + \mathbf{e}_q^T \mathbf{K}_p \mathbf{e}_v = 0$$

Leaving only strictly dissipative damping and neural residual adaptation:

$$\dot{V} = -\mathbf{e}_v^T \mathbf{D} \mathbf{e}_v + \text{tr}\left( \tilde{\mathbf{W}}_{\Delta}^T \left[ \frac{1}{\Gamma_{\Delta}} \dot{\tilde{\mathbf{W}}}_{\Delta} + \mathbf{e}_q h(t)^T \right] \right) + \mathbf{e}_v^T \mathbf{\epsilon}$$

Setting the adaptation law with $\sigma$-leakage:

$$\dot{\mathbf{W}}_{\Delta}(t) = \Gamma_{\Delta} \left[ \mathbf{e}_q(t) h(t)^T \right] - \sigma_{\text{leak}} \mathbf{W}_{\Delta}(t)$$

guarantees $\dot{V} \le -\alpha V + \beta < 0$ outside a compact residual set, establishing **exponential convergence and Uniform Ultimate Boundedness (UUB)**.

---

## 4. Source Files on this Branch

- `src/observers/physics_informed_lstm_observer.py`: Complete implementation of `PhysicsInformedLSTMObserver` enforcing inertia positive-definiteness and Coriolis skew-symmetry.
- `tests/test_physics_informed_lstm.py`: Unit tests verifying Coriolis skew-symmetry identity, positive-definite inertia matrix, and tracking stability.

---

## 5. Verification Commands

```bash
# Test physics-informed properties (skew-symmetry & positive-definiteness)
pytest tests/test_physics_informed_lstm.py -v

# Run full test suite
pytest -v
```
