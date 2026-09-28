#!/usr/bin/env python3
"""
Run estimation benchmarks on the Feedback 33-936S rig and print comparison table.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.simulation.benchmark_runner import SimulationConfig, run_benchmark_comparison


def main() -> None:
    print("=" * 80)
    print("FEEDBACK INSTRUMENTS 33-936S CART-INVERTED PENDULUM BENCHMARK TESTBED")
    print("Running Baseline Estimator Comparisons under PCI-1711 DAQ Noise & Stiction...")
    print("=" * 80)

    cfg = SimulationConfig(t_final=5.0, dt=0.001, excitation_mode="lqr_stabilize")
    result = run_benchmark_comparison(cfg)

    print("\n[+] Benchmark Completed across 4 Estimators (5000 time steps @ 1 kHz):")
    print("    - True plant states: x, x_dot, theta, theta_dot")
    print("    - Measurements: Optical Encoders with uniform noise & quantization")
    print("    - Non-linear effects: Asymmetric stiction deadband [+0.15V, -0.12V]\n")

    summary_rows = []
    for obs_name, report in result.reports.items():
        df = report.summary_df
        summary_rows.append({
            "Estimator": obs_name,
            "Cart_Pos_RMSE [m]": df.loc["x", "RMSE"],
            "Cart_Vel_RMSE [m/s]": df.loc["x_dot", "RMSE"],
            "Theta_RMSE [rad]": df.loc["theta", "RMSE"],
            "Theta_Dot_RMSE [rad/s]": df.loc["theta_dot", "RMSE"],
            "Vel_Chattering_Ratio": df.loc["x_dot", "Chattering_Ratio"],
            "AngVel_Chattering_Ratio": df.loc["theta_dot", "Chattering_Ratio"],
        })

    comp_df = pd.DataFrame(summary_rows).set_index("Estimator")
    print(comp_df.to_string())
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
