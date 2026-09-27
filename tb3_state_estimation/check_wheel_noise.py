#!/usr/bin/env python3
"""
Check A: WheelOdomCorruptor (live relay core) vs corrupt_odometry()
(offline reference) on a recorded bag. No ROS nodes involved.

corrupt_odometry() draws all right-wheel noise first, then all left-wheel
noise, from one generator. Pre-drawing two arrays in that same order from
the same seed gives the class identical samples, so the comparison covers
the noise path too, not only the deterministic part.

Prediction (written before running): every max difference is exactly 0.0.
Same inputs, same operations, same order. Tiny nonzero values (~1e-15)
would mean an operation-order difference; anything above 1e-9 is a bug.

Run from the same folder as test_noisy_odom.py:
    python3 check_wheel_noise.py ~/ekf_long_test
"""

import argparse

import numpy as np

from test_noisy_odom import PX, PY, T, V, W, YAW, corrupt_odometry, load_bag
from tb3_state_estimation.differential_drive_ekf import wrap_angle
from tb3_state_estimation.wheel_noise import WheelOdomCorruptor

TOL = 1e-9


def run_class(odom, b, mismatch, c, n_r, n_l):
    """Feed the bag through the class message by message, as the node will."""
    corr = WheelOdomCorruptor(wheel_separation=b, mismatch=mismatch, common_scale=c)
    n = len(odom)
    v_n = np.empty(n)
    track = np.empty((n, 3))

    corr.initialize(odom[0, T], odom[0, PX], odom[0, PY], odom[0, YAW])
    v_n[0], _ = corr.corrupt_twist(odom[0, V], odom[0, W], n_r[0], n_l[0])
    track[0] = corr.pose
    for k in range(1, n):
        v_n[k], _, _ = corr.step(odom[k, T], odom[k, V], odom[k, W], n_r[k], n_l[k])
        track[k] = corr.pose
    return v_n, track


def compare(odom, b, mismatch, c, sigma, seed):
    rng = np.random.default_rng(seed)
    n_r = rng.normal(0.0, sigma, len(odom))
    n_l = rng.normal(0.0, sigma, len(odom))

    v_ref, track_ref = corrupt_odometry(
        odom, b, +mismatch / 2.0, -mismatch / 2.0, sigma,
        np.random.default_rng(seed), c=c)
    v_cls, track_cls = run_class(odom, b, mismatch, c, n_r, n_l)

    return {
        "v [m/s]": float(np.max(np.abs(v_cls - v_ref))),
        "pos [m]": float(np.max(np.hypot(track_cls[:, 0] - track_ref[:, 0],
                                         track_cls[:, 1] - track_ref[:, 1]))),
        "heading [rad]": float(np.max(np.abs(wrap_angle(track_cls[:, 2] - track_ref[:, 2])))),
    }


def run_cases(odom, b):
    cases = [
        # (label, mismatch, common_scale, sigma, seed)
        ("deterministic path", 0.005, 0.01, 0.0, 0),
        ("default noise, seed 0", 0.005, 0.0, 0.03, 0),
        ("noise + common scale, seed 7", 0.005, 0.01, 0.03, 7),
    ]
    all_ok = True
    for label, m, c, sigma, seed in cases:
        diffs = compare(odom, b, m, c, sigma, seed)
        ok = all(d <= TOL for d in diffs.values())
        all_ok &= ok
        detail = ", ".join(f"{k} {d:.1e}" for k, d in diffs.items())
        print(f"{'PASS' if ok else 'FAIL'}  {label:<30} max diff: {detail}")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--b", type=float, default=0.160)
    args = ap.parse_args()

    odom, _ = load_bag(args.bag)
    print(f"Loaded {len(odom)} odom messages")
    ok = run_cases(odom, args.b)
    print("Check A passed." if ok else "Check A FAILED.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
