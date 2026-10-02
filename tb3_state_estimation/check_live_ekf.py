#!/usr/bin/env python3
"""
Stage 4 live, step 3: checks for the live EKF node on a bag recorded with
/odom, /odom_noisy, /imu and /odometry/filtered.

    E1a  the node's first recorded pose equals /odom_noisy's pose at the
         same stamp: the node runs in the noisy odometry's frame
    E1   the node equals the offline model: run_filter() on the bag's own
         /odom_noisy and /imu, started from the node's first recorded
         state, compared stamp by stamp with /odometry/filtered
    E2   /odom_noisy and the node against clean /odom: the offline
         experiment's metric, for one live noise realization

Message order. Odom and IMU share a stamp at every odom tick (the script
reports how often). run_filter() puts odom first on equal stamps; the node
sees them in arrival order. /odom_noisy reaches the node through the relay,
one hop after /imu, so the expected live order is IMU first. E1 replays
both orders. The node should match one of them closely; the other should
differ by up to about |omega| * 5 ms (one IMU period of time shift), about
0.3 deg during a 1 rad/s spin. That split is itself the evidence of which
order the node followed.

Tolerances, fixed before running:
    E1a  |position| < 1e-6 m, |heading| < 0.05 deg (the heading part is the
         gyro random walk between node start and recording start)
    E1   matching order: max |heading diff| < 0.05 deg,
         max position diff < 1 mm
         (not zero: under CPU load an occasional tick can be processed in
         the other order, each costing about |d omega/dt| * 5 ms * 20 ms)

The node started before the recorder, so E1 starts run_filter() from the
node's own first recorded state (same trick as check_relay_bag.py's B1).
Because of that, E1 cannot see an initialization error; E1a covers it.
Honest limit: the robot spawns at the odom origin, so an initial pose of
zero and an initial pose from the message look the same here. The node's
"Initialized ..." log line is the direct evidence for that part.

Run from the same folder as test_noisy_odom.py:
    python3 check_live_ekf.py ~/bags/ekf_gt_run1 [--until <s after t0>] [--plot] [--save PNG]
"""

import argparse

import numpy as np

from compare_ground_truth import anchor_time, track_style
from test_noisy_odom import PX, PY, T, V, W, YAW, errors, load_bag, rms, run_filter
from tb3_state_estimation.differential_drive_ekf import wrap_angle

E1_HEADING_DEG = 0.05
E1_POS_M = 0.001
E1A_POS_M = 1e-6
E1A_HEADING_DEG = 0.05
STILL_V = 1e-3   # [m/s]
STILL_W = 1e-3   # [rad/s]
ORDERS = (("odom first", False), ("IMU first", True))
# Replay orders are not tracks, so they deliberately avoid the track colours
# (black, blue, red, green) used in every other figure.
ORDER_COLORS = ("tab:orange", "tab:purple")


def verdict(ok):
    return "PASS" if ok else "FAIL"


def aligned(a, b):
    """Row indices (i in a, j in b) of stamps present in both, in a's order.
    Stamps are copied bit-identically through relay and node, and converted
    by the same expression, so exact float equality is the right test."""
    b_index = {t: j for j, t in enumerate(b[:, T])}
    pairs = [(i, b_index[t]) for i, t in enumerate(a[:, T]) if t in b_index]
    if not pairs:
        raise SystemExit("No common stamps between the two topics.")
    i, j = zip(*pairs)
    return np.array(i), np.array(j)


def check_e1a(noisy, filt):
    k, j = aligned(noisy, filt)
    k, j = k[0], j[0]
    dp = float(np.hypot(filt[j, PX] - noisy[k, PX], filt[j, PY] - noisy[k, PY]))
    dh = float(np.degrees(abs(wrap_angle(filt[j, YAW] - noisy[k, YAW]))))
    ok = dp < E1A_POS_M and dh < E1A_HEADING_DEG
    print(f"{verdict(ok)}  E1a first common stamp: |dpos| {dp:.1e} m, "
          f"|dheading| {dh:.4f} deg (budget {E1A_POS_M:.0e} m, {E1A_HEADING_DEG} deg)")
    return ok


def replay(noisy, imu, filt, k, j, imu_first):
    """Offline filter from the node's first recorded state [x, y, yaw,
    omega]; v stays /odom_noisy's, exactly as the node uses it."""
    k0, j0 = k[0], j[0]
    ref_in = noisy[k0:].copy()
    ref_in[0, [PX, PY, YAW, W]] = filt[j0, [PX, PY, YAW, W]]
    ref, _ = run_filter(ref_in, ref_in[:, V], imu, imu_first=imu_first)
    kk = k - k0
    d_head = wrap_angle(filt[j, YAW] - ref[kk, 2])
    d_pos = np.hypot(filt[j, PX] - ref[kk, 0], filt[j, PY] - ref[kk, 1])
    return d_head, d_pos


def check_e1(noisy, imu, filt):
    k, j = aligned(noisy, filt)
    k0 = k[0]

    imu_stamps = set(imu[:, T])
    shared = np.mean([t in imu_stamps for t in noisy[k, T]])
    missed = (len(noisy) - k0) - len(k)
    orphans = len(filt) - len(j)
    print(f"INFO  E1  {len(k)} stamps compared; {100 * shared:.1f}% of odom ticks share "
          f"a stamp with an IMU message")
    print(f"INFO  E1  /odom_noisy with no node output: {missed}; node output with no "
          f"recorded /odom_noisy: {orphans} (a few at the recording edges is normal)")

    omega = filt[j, W]
    still = (np.abs(noisy[k, V]) < STILL_V) & (np.abs(omega) < STILL_W)
    res = {}
    for label, imu_first in ORDERS:
        d_head, d_pos = replay(noisy, imu, filt, k, j, imu_first)
        res[label] = (d_head, d_pos)
        still_max = np.degrees(np.abs(d_head[still]).max()) if still.any() else float("nan")
        print(f"INFO  E1  replay {label:<11} heading diff max {np.degrees(np.abs(d_head).max()):.4f} deg "
              f"(while stopped {still_max:.4f}), position diff max {1000 * d_pos.max():.3f} mm, "
              f"final {1000 * d_pos[-1]:.3f} mm")

    best = min(res, key=lambda lab: np.abs(res[lab][0]).max())
    d_head, d_pos = res[best]
    h, p = float(np.degrees(np.abs(d_head).max())), float(d_pos.max())
    ok = h < E1_HEADING_DEG and p < E1_POS_M
    print(f"{verdict(ok)}  E1  node matches the '{best}' replay: max heading diff {h:.4f} deg "
          f"(budget {E1_HEADING_DEG}), max position diff {1000 * p:.3f} mm "
          f"(budget {1000 * E1_POS_M:.0f} mm)")
    return ok, {"t": noisy[k, T], "res": res, "best": best}


def report_e2(clean, noisy, filt):
    k, j = aligned(noisy, filt)
    c_index = {t: i for i, t in enumerate(clean[:, T])}
    keep = np.array([t in c_index for t in noisy[k, T]])
    k, j = k[keep], j[keep]
    c = clean[[c_index[t] for t in noisy[k, T]]]

    print(f"INFO  E2  {len(k)} stamps common to /odom, /odom_noisy and /odometry/filtered")
    print(f"      {'track':<20}{'final |hdg|':>12}{'max |hdg|':>11}{'final pos':>11}"
          f"{'max pos':>10}{'RMS along':>11}{'RMS cross':>11}")
    print(f"      {'':<20}{'[deg]':>12}{'[deg]':>11}{'[mm]':>11}{'[mm]':>10}"
          f"{'[mm]':>11}{'[mm]':>11}")
    out = {"t": noisy[k, T]}
    for name, track in (("/odom_noisy", noisy[k]), ("/odometry/filtered", filt[j])):
        e = errors(track[:, [PX, PY, YAW]], c)
        out[name] = e
        print(f"      {name:<20}{np.degrees(abs(e['heading'][-1])):>12.4f}"
              f"{np.degrees(np.abs(e['heading']).max()):>11.4f}"
              f"{1000 * e['pos'][-1]:>11.2f}{1000 * e['pos'].max():>10.2f}"
              f"{1000 * rms(e['along']):>11.2f}{1000 * rms(e['cross']):>11.2f}")
    h0 = np.degrees(out["/odometry/filtered"]["heading"][0])
    print(f"      EKF heading error at the first sample {h0:+.4f} deg "
          "(stationary gyro random walk before recording; not removed here)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--until", type=float, default=None,
                    help="analyse only up to this many seconds after t0, "
                         "the same t0 as compare_ground_truth.py")
    ap.add_argument("--plot", action="store_true", help="show the figure in a window")
    ap.add_argument("--save", metavar="PNG", default=None,
                    help="write the figure to this file (no window unless --plot)")
    args = ap.parse_args()

    clean, _ = load_bag(args.bag, odom_topic="/odom")
    noisy, imu = load_bag(args.bag, odom_topic="/odom_noisy")
    filt, _ = load_bag(args.bag, odom_topic="/odometry/filtered")
    print(f"/odom {len(clean)}, /odom_noisy {len(noisy)}, "
          f"/odometry/filtered {len(filt)}, /imu {len(imu)} messages")

    t0 = anchor_time(clean)
    if args.until is not None:
        cut = t0 + args.until
        clean = clean[clean[:, T] <= cut]
        noisy = noisy[noisy[:, T] <= cut]
        filt = filt[filt[:, T] <= cut]
        print(f"[window: t - t0 <= {args.until:.1f} s]")

    ok_a = check_e1a(noisy, filt)
    ok_1, e1 = check_e1(noisy, imu, filt)
    e2 = report_e2(clean, noisy, filt)
    ok = ok_a and ok_1
    print("E1 checks passed." if ok else "Some E1 checks FAILED.")

    if args.plot or args.save:
        import matplotlib
        if not args.plot:
            matplotlib.use("Agg")  # file output only, no window needed
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        t1 = e1["t"] - t0
        for (label, _), col in zip(ORDERS, ORDER_COLORS):
            d_head, d_pos = e1["res"][label]
            ax[0].plot(t1, np.degrees(d_head), color=col, label=f"node - replay ({label})")
            ax[1].plot(t1, 1000 * d_pos, color=col, label=f"node - replay ({label})")
        ax[0].set_title("E1: live node minus offline replay, both orders for equal stamps")
        ax[0].set_ylabel("heading diff [deg]")
        ax[0].legend(fontsize="small")
        ax[1].set_ylabel("position diff [mm]")
        ax[1].legend(fontsize="small")
        t2 = e2["t"] - t0
        for topic in ("/odom_noisy", "/odometry/filtered"):
            lab, col, _ = track_style(topic)
            ax[2].plot(t2, np.degrees(e2[topic]["heading"]), color=col, label=lab)
        ax[2].set_title("E2: heading error against clean odometry")
        ax[2].set_ylabel("heading error [deg]")
        ax[2].set_xlabel("t - t0 [s]")
        ax[2].legend(fontsize="small")
        plt.tight_layout()
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"saved {args.save}")
        if args.plot:
            plt.show()

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
