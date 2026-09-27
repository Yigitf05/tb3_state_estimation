#!/usr/bin/env python3
"""
Checks B, C, D for the live relay, on a bag recorded with /odom,
/odom_noisy and /imu. The relay config is read from relay_params.yaml,
which you copy into the bag folder after recording.

    C  (always)          /odom_noisy's pose is the integral of its own twist
    B  (all noise zero)  /odom_noisy reproduces /odom: twist exactly, pose
                         to the Gazebo-vs-Euler integration residual
    D1 (mismatch, sigma=0)  heading drift matches the deterministic prediction
    D2 (sigma > 0)       normalized twist noise has std 1.00, and heading
                         error lies within +-2 sigma of the prediction

Messages are matched by header.stamp. The relay copies stamps, so they are
bit-identical. A clean stamp with no noisy partner means the relay missed
that message (best-effort subscription).

Run from the same folder as test_noisy_odom.py:
    python3 check_relay_bag.py ~/relay_zero
"""

import argparse
import os

import numpy as np
import yaml

from test_noisy_odom import MAX_DT_SEC, PX, PY, T, V, W, YAW, corrupt_odometry, load_bag
from tb3_state_estimation.differential_drive_ekf import wrap_angle

# Budget for the Gazebo-vs-Euler integration residual (earlier measurement:
# ~1.5 mm, ~0.1 deg on the long bag). Used by B and D1.
RESIDUAL_POS_M = 0.01
RESIDUAL_HEADING_DEG = 0.2


def verdict(ok):
    return "PASS" if ok else "FAIL"


def read_params(path):
    with open(path) as f:
        return yaml.safe_load(f)["odom_noise_relay"]["ros__parameters"]


def align(clean, noisy):
    """Keep only clean messages the relay also published, matched by stamp."""
    clean_index = {t: i for i, t in enumerate(clean[:, T])}
    noisy_stamps = set(noisy[:, T])
    missing = sum(1 for t in clean[:, T] if t not in noisy_stamps)
    pairs = [(clean_index[t], j) for j, t in enumerate(noisy[:, T]) if t in clean_index]
    orphans = len(noisy) - len(pairs)
    ci, nj = zip(*pairs)
    return clean[list(ci)], noisy[list(nj)], missing, orphans


def integrated_steps(t, max_dt):
    dt = np.diff(t)
    return dt, (dt > 0.0) & (dt <= max_dt)


def check_c(noisy, b):
    """Integrate /odom_noisy's own twist with zero noise. The split/recombine
    inside corrupt_odometry adds a few ulps per step, so expect ~1e-13, not 0."""
    _, track = corrupt_odometry(noisy, b, 0.0, 0.0, 0.0, np.random.default_rng(0))
    pos = float(np.max(np.hypot(track[:, 0] - noisy[:, PX], track[:, 1] - noisy[:, PY])))
    head = float(np.max(np.abs(wrap_angle(track[:, 2] - noisy[:, YAW]))))
    ok = pos < 1e-9 and head < 1e-9
    print(f"{verdict(ok)}  C  self-consistency: max pos {pos:.1e} m, max heading {head:.1e} rad")
    return ok


def check_b(clean, noisy, b):
    """B1 (pass/fail): the relay equals its specification, i.e. Euler
    integration of the clean twist, on the same matched messages.
    B2 (information): distance from Gazebo's own pose. That measures
    Gazebo's integrator versus Euler, not the relay."""
    dv = float(np.max(np.abs(noisy[:, V] - clean[:, V])))
    dw = float(np.max(np.abs(noisy[:, W] - clean[:, W])))
    ok_twist = dv < 1e-12 and dw < 1e-12
    print(f"{verdict(ok_twist)}  B1 twist: max |dv| {dv:.1e} m/s, max |dw| {dw:.1e} rad/s")

    # Start the reference from the relay's first recorded pose: the relay may
    # have been started before recording, so the two can differ at t0 by the
    # residual accumulated before the bag began.
    ref_in = clean.copy()
    ref_in[0, [PX, PY, YAW]] = noisy[0, [PX, PY, YAW]]
    _, ref = corrupt_odometry(ref_in, b, 0.0, 0.0, 0.0, np.random.default_rng(0))
    spec_pos = float(np.max(np.hypot(ref[:, 0] - noisy[:, PX], ref[:, 1] - noisy[:, PY])))
    spec_head = float(np.max(np.abs(wrap_angle(ref[:, 2] - noisy[:, YAW]))))
    ok_spec = spec_pos < 1e-9 and spec_head < 1e-9
    print(f"{verdict(ok_spec)}  B1 pose vs Euler of clean twist: max {spec_pos:.1e} m, "
          f"{spec_head:.1e} rad")

    pos = np.hypot(noisy[:, PX] - clean[:, PX], noisy[:, PY] - clean[:, PY])
    head = np.degrees(np.abs(wrap_angle(noisy[:, YAW] - clean[:, YAW])))
    note = ("as expected" if pos.max() < RESIDUAL_POS_M and head.max() < RESIDUAL_HEADING_DEG
            else "larger than the offline residual, investigate")
    print(f"INFO  B2 pose vs Gazebo's pose: max {1000 * pos.max():.2f} mm, "
          f"max {head.max():.3f} deg ({note})")
    return ok_twist and ok_spec


def check_d(clean, noisy, p):
    b, m, c, sigma = p["wheel_separation"], p["mismatch"], p["common_scale"], p["sigma"]
    s_r, s_l = m / 2.0, -m / 2.0
    v, w = clean[:, V], clean[:, W]
    a = w * b / 2.0
    dt, step = integrated_steps(clean[:, T], p["max_dt"])

    # Deterministic part: w_n - w = c*w + (1+c)*m*v/b. Integrated over the
    # same steps the relay integrated (current twist over preceding dt).
    dw_det = c * w[1:] + (1.0 + c) * m * v[1:] / b
    predicted = float(np.sum(dw_det[step] * dt[step]))

    # Random walk from white wheel noise: per-step heading increment variance
    # (1+c)^2 sigma^2 dt^2 ((v_r(1+s_r))^2 + (v_l(1+s_l))^2) / b^2.
    v_r, v_l = (v + a)[1:], (v - a)[1:]
    var_step = ((1.0 + c) * sigma * dt / b) ** 2 * ((v_r * (1 + s_r)) ** 2 + (v_l * (1 + s_l)) ** 2)
    sd_rw = float(np.sqrt(np.sum(var_step[step])))

    measured = float(wrap_angle(noisy[-1, YAW] - clean[-1, YAW]))
    signed_d = float(np.sum(v[1:][step] * dt[step]))
    print(f"   D  signed distance {signed_d:.3f} m, predicted heading error "
          f"{np.degrees(predicted):.3f} deg, random-walk sd {np.degrees(sd_rw):.3f} deg")
    print(f"   D  measured final heading error {np.degrees(measured):.3f} deg")

    oks = []
    if sigma == 0.0:
        err = abs(np.degrees(measured - predicted))
        ok = err < RESIDUAL_HEADING_DEG
        print(f"{verdict(ok)}  D1 |measured - predicted| = {err:.3f} deg "
              f"(budget {RESIDUAL_HEADING_DEG} deg)")
        oks.append(ok)
    else:
        band = 2.0 * sd_rw + np.radians(RESIDUAL_HEADING_DEG)
        ok = abs(measured - predicted) <= band
        print(f"{verdict(ok)}  D2 heading within prediction +- (2 sd + residual) = "
              f"+-{np.degrees(band):.2f} deg (one run: expect a fail ~5% of the time)")
        oks.append(ok)

        # Twist noise: subtract the known deterministic part, divide by the
        # predicted per-message std. Result should be ~N(0, 1).
        v_det = (1.0 + c) * ((v + a) * (1 + s_r) + (v - a) * (1 + s_l)) / 2.0
        sd_v = (1.0 + c) * sigma * np.sqrt(((v + a) * (1 + s_r)) ** 2
                                           + ((v - a) * (1 + s_l)) ** 2) / 2.0
        moving = sd_v > 1e-6
        z = (noisy[moving, V] - v_det[moving]) / sd_v[moving]
        n = int(moving.sum())
        se = 1.0 / np.sqrt(2.0 * n)            # standard error of a sample std
        ok = abs(z.std() - 1.0) < 3.0 * se and abs(z.mean()) < 3.0 / np.sqrt(n)
        print(f"{verdict(ok)}  D2 normalized twist noise over {n} moving samples: "
              f"std {z.std():.4f} (expect 1 +- {3 * se:.4f}), mean {z.mean():+.4f}")
        oks.append(ok)
    return all(oks)


def run_checks(clean, noisy, p):
    clean_a, noisy_a, missing, orphans = align(clean, noisy)
    print(f"{len(clean)} clean, {len(noisy)} noisy messages; matched {len(clean_a)}, "
          f"relay missed {missing}, noisy without clean partner {orphans}")
    print("Config: " + ", ".join(f"{k}={v}" for k, v in p.items()))

    oks = [check_c(noisy, p["wheel_separation"])]
    if p["mismatch"] == 0.0 and p["sigma"] == 0.0 and p["common_scale"] == 0.0:
        oks.append(check_b(clean_a, noisy_a, p["wheel_separation"]))
    else:
        oks.append(check_d(clean_a, noisy_a, p))
    print("All checks passed." if all(oks) else "Some checks FAILED.")
    return all(oks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--params", default=None,
                    help="relay params yaml (default: <bag>/relay_params.yaml)")
    args = ap.parse_args()

    params_path = args.params or os.path.join(args.bag, "relay_params.yaml")
    p = read_params(params_path)
    p.setdefault("max_dt", MAX_DT_SEC)
    p.setdefault("common_scale", 0.0)

    clean, _ = load_bag(args.bag)
    noisy, _ = load_bag(args.bag, odom_topic="/odom_noisy")
    raise SystemExit(0 if run_checks(clean, noisy, p) else 1)


if __name__ == "__main__":
    main()
