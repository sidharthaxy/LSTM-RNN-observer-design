#!/usr/bin/env python3
"""
Generate publication-quality comparison figures for baseline estimators on Feedback 33-936S rig.
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.simulation.benchmark_runner import SimulationConfig, run_benchmark_comparison


def main() -> None:
    print("[*] Running simulation for visualization...")
    cfg = SimulationConfig(t_final=4.0, dt=0.001, excitation_mode="lqr_stabilize")
    result = run_benchmark_comparison(cfg)

    t = result.t
    true_s = result.state_true
    estimates = result.estimates

    fig_dir = ROOT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    titles = [
        "Cart Position $x$ [m]",
        "Cart Velocity $\\dot{x}$ [m/s] (Unmeasured Ground Truth vs Estimators)",
        "Pendulum Angle $\\theta$ [rad]",
        "Pendulum Angular Velocity $\\dot{\\theta}$ [rad/s] (Unmeasured Ground Truth vs Estimators)"
    ]

    colors = {
        "Ground Truth": "black",
        "Dirty_Derivative": "#e74c3c",
        "Butterworth_Diff": "#e67e22",
        "Continuous_EKF": "#2980b9",
        "Shallow_Dynamic_RNN": "#27ae60",
    }

    # Plot True States
    for idx in range(4):
        axes[idx].plot(t, true_s[:, idx], label="Ground Truth", color=colors["Ground Truth"], linewidth=2.0, zorder=5)

    # Plot Estimators
    for name, est in estimates.items():
        c = colors.get(name, "#8e44ad")
        ls = "--" if "EKF" in name or "RNN" in name else ":"
        alpha = 0.85
        for idx in range(4):
            axes[idx].plot(t, est[:, idx], label=name, color=c, linestyle=ls, alpha=alpha, linewidth=1.2)

    for idx, ax in enumerate(axes):
        ax.set_ylabel(titles[idx], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper right", ncol=3, frameon=True, fontsize=9)

    axes[-1].set_xlabel("Time [s]", fontsize=11, fontweight="bold")
    plt.suptitle("Feedback Instruments 33-936S: Baseline Velocity Reconstruction Under Optical Noise & Stiction", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = fig_dir / "baseline_estimation_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"[+] Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
