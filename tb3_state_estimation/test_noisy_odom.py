#!/usr/bin/env python3
"""
Stage 4a, offline experiment: wheel-slip noise injection.

Takes a recorded bag of clean /odom + /imu, corrupts the odometry at the
WHEEL level, and compares three tracks against ground truth:

    truth     clean /odom pose (valid stand-in only while Gazebo produces
              no physical slip; checked against real ground truth in the
              live stage)
    raw       dead reckoning from the corrupted wheel speeds
    filtered  DifferentialDriveEKF: corrupted v as control input, real
              /imu gyro as the omega measurement

Noise model (agreed design, per wheel):
    v_wheel_noisy = v_wheel * (1 + c) * (1 + s) * (1 + n_k)
    c    common scale error on BOTH wheels (wrong nominal wheel radius).
         Shifts distance travelled, not heading. Default 0.
    s    differential scale error, fixed for the whole run (wheel diameter
         mismatch, Borenstein & Feng 1996). s_r = +m/2, s_l = -m/2.
    n_k  white multiplicative noise per odom message, std sigma.

Pose and twist of the "raw" track stay consistent by construction: the
raw pose is the integral of the corrupted twist, as on a real robot.

Gyro fairness (optional): --gyro-bias adds a constant bias [deg/s] to the
/imu yaw rate before the filter sees it. The filter has no bias state, so
its heading drifts by bias * elapsed time, even while stationary. Wheel
drift grows with distance instead. --sweep runs a range of biases and
prints where the EKF stops beating raw odometry.

Prediction written down before running (see project notes):
    - filtered beats raw on heading, and therefore on cross-track and
      total position error
    - filtered does NOT beat raw on along-track error (it uses odom's v)
    - P's x/y variance grows in both cases (x, y unobservable)

The predict/update sequencing, dt source (header.stamp) and dt guards
match ekf_node.py, so this is a faithful predictor of the node. The one
difference is the order of an odom and an IMU message with the same stamp,
see run_filter().

Usage:
    python3 test_noisy_odom.py <bag_dir> [--seeds 20] [--mismatch 0.005]
                               [--sigma 0.03] [--common-scale 0.0]
                               [--gyro-bias 0.0] [--sweep]
                               [--b 0.160] [--plot] [--save PNG]
"""

import argparse

import numpy as np

from tb3_state_estimation.differential_drive_ekf import (
    DifferentialDriveEKF,
    quaternion_to_yaw,
    wrap_angle,
)

# Same guard values as ekf_node.py
EXPECTED_ODOM_PERIOD_SEC = 1.0 / 50.0
MAX_DT_SEC = 5.0 * EXPECTED_ODOM_PERIOD_SEC

# Column indices of the odom array built by load_bag()
T, V, W, PX, PY, YAW = range(6)


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def load_bag(path, odom_topic="/odom", imu_topic="/imu"):
    """Read the bag and return:
        odom: (N, 6) array [t, v, omega, x, y, yaw]
        imu:  (M, 2) array [t, gyro_z]
    Times come from each message's own header.stamp, NOT the bag's
    receive timestamp (the gotcha from compare_live_ekf.py)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    # Empty storage_id lets rosbag2 detect mcap / sqlite3 from metadata.yaml
    reader.open(
        rosbag2_py.StorageOptions(uri=path, storage_id=""),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for topic in (odom_topic, imu_topic):
        if topic not in types:
            raise SystemExit(f"Topic {topic} not in bag. Found: {sorted(types)}")
    odom_cls = get_message(types[odom_topic])
    imu_cls = get_message(types[imu_topic])

    odom, imu = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == odom_topic:
            m = deserialize_message(data, odom_cls)
            q = m.pose.pose.orientation
            odom.append((
                stamp_to_sec(m.header.stamp),
                m.twist.twist.linear.x,
                m.twist.twist.angular.z,
                m.pose.pose.position.x,
                m.pose.pose.position.y,
                quaternion_to_yaw(q.x, q.y, q.z, q.w),
            ))
        elif topic == imu_topic:
            m = deserialize_message(data, imu_cls)
            imu.append((stamp_to_sec(m.header.stamp), m.angular_velocity.z))

    # Sort by header stamp, since bag order is arrival order
    return np.array(sorted(odom)), np.array(sorted(imu))


def corrupt_odometry(odom, b, s_r, s_l, sigma, rng, c=0.0):
    """Wheel-level corruption. Returns the noisy v (the filter's control
    input) and the raw dead-reckoning track (N, 3) [x, y, yaw]."""
    t, v, w = odom[:, T], odom[:, V], odom[:, W]

    v_r = v + w * b / 2.0
    v_l = v - w * b / 2.0
    v_r_n = v_r * (1.0 + c) * (1.0 + s_r) * (1.0 + rng.normal(0.0, sigma, len(t)))
    v_l_n = v_l * (1.0 + c) * (1.0 + s_l) * (1.0 + rng.normal(0.0, sigma, len(t)))

    v_n = (v_r_n + v_l_n) / 2.0
    w_n = (v_r_n - v_l_n) / b

    # Integrate with the same Euler step and dt guards the filter uses,
    # so any difference between raw and filtered comes from the gyro,
    # not from a different integration scheme.
    track = np.empty((len(t), 3))
    track[0] = odom[0, [PX, PY, YAW]]
    for k in range(1, len(t)):
        x, y, th = track[k - 1]
        dt = t[k] - t[k - 1]
        if 0.0 < dt <= MAX_DT_SEC:
            x += v_n[k] * np.cos(th) * dt
            y += v_n[k] * np.sin(th) * dt
            th = wrap_angle(th + w_n[k] * dt)
        track[k] = (x, y, th)
    return v_n, track


def run_filter(odom, v_control, imu, imu_first=False):
    """Replay odom (predict) and imu (update) events in header-stamp order,
    as ekf_node.py receives them. Returns track (N, 3) and the full 2x2
    x/y covariance block (N, 2, 2) at each odom message.

    Odom and IMU share a stamp at every odom tick. Default: odom first on
    equal stamps. This is the offline experiment's convention and all its
    published results use it. imu_first=True applies the IMU sample first,
    the order the live node is expected to see, since its odometry arrives
    through the relay one hop after the IMU (check_live_ekf.py compares
    both orders)."""
    ekf = DifferentialDriveEKF(
        initial_state=[odom[0, PX], odom[0, PY], odom[0, YAW], odom[0, W]],
        # Odom frame is defined by the start pose: position and heading are
        # known essentially exactly at t0. Omega starts from odom's own rate,
        # which the gyro corrects within a step anyway.
        initial_covariance=np.diag([1e-8, 1e-8, 1e-8, 1e-3]),
    )

    # (time, kind, index). On equal stamps the lower kind is processed first.
    k_odom, k_imu = (1, 0) if imu_first else (0, 1)
    t_start = odom[0, T]
    events = [(t, k_odom, k) for k, t in enumerate(odom[:, T])]
    # IMU stamped before the first odom message is dropped, as in the node.
    # With imu_first, an IMU message stamped exactly at the first odom
    # message would reach the node before initialization, so it is dropped
    # too. With the default this is the original t >= t_start.
    events += [(t, k_imu, j) for j, t in enumerate(imu[:, T])
               if t > t_start or (t == t_start and not imu_first)]
    events.sort()

    track = np.empty((len(odom), 3))
    P_xy = np.empty((len(odom), 2, 2))
    track[0] = ekf.pose
    P_xy[0] = ekf.P[:2, :2]
    last_t = odom[0, T]

    for t, kind, idx in events:
        if kind == k_imu:
            ekf.update_gyro(imu[idx, 1])
            continue
        if idx == 0:
            continue
        dt = t - last_t
        last_t = t
        if 0.0 < dt <= MAX_DT_SEC:
            ekf.predict(v=v_control[idx], dt=dt)
        track[idx] = ekf.pose
        P_xy[idx] = ekf.P[:2, :2]

    return track, P_xy


def errors(track, odom):
    """Errors against truth, with position error split into along-track
    and cross-track components in truth's own heading frame."""
    ex = track[:, 0] - odom[:, PX]
    ey = track[:, 1] - odom[:, PY]
    th = odom[:, YAW]
    return {
        "heading": wrap_angle(track[:, 2] - th),
        "pos": np.hypot(ex, ey),
        "along": ex * np.cos(th) + ey * np.sin(th),
        "cross": -ex * np.sin(th) + ey * np.cos(th),
    }


def sigma_along_cross(P_xy, odom):
    """Rotate each 2x2 position covariance into truth's heading frame, so
    the filter's claimed uncertainty can be compared component by component
    with the along-track and cross-track errors."""
    th = odom[:, YAW]
    c, s = np.cos(th), np.sin(th)
    # Rows of the rotation: along = [c, s], cross = [-s, c]
    var_along = c * c * P_xy[:, 0, 0] + 2 * c * s * P_xy[:, 0, 1] + s * s * P_xy[:, 1, 1]
    var_cross = s * s * P_xy[:, 0, 0] - 2 * c * s * P_xy[:, 0, 1] + c * c * P_xy[:, 1, 1]
    return np.sqrt(np.maximum(var_along, 0.0)), np.sqrt(np.maximum(var_cross, 0.0))


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--mismatch", type=float, default=0.005,
                    help="wheel diameter mismatch m; s_r=+m/2, s_l=-m/2")
    ap.add_argument("--sigma", type=float, default=0.03,
                    help="white multiplicative noise std per wheel per message")
    ap.add_argument("--common-scale", type=float, default=0.0,
                    help="common scale error c on both wheels (e.g. 0.01 = 1%%)")
    ap.add_argument("--b", type=float, default=0.160,
                    help="wheel separation [m]; check against your model's params")
    ap.add_argument("--gyro-bias", type=float, default=0.0,
                    help="constant gyro yaw-rate bias added to /imu [deg/s]")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep gyro bias values and print the crossover table")
    ap.add_argument("--plot", action="store_true", help="show the figure in a window")
    ap.add_argument("--save", metavar="PNG", default=None,
                    help="write the seed-0 figure to this file (no window unless --plot)")
    args = ap.parse_args()

    odom, imu = load_bag(args.bag)
    duration = odom[-1, T] - odom[0, T]
    dt_all = np.diff(odom[:, T])
    path_len = float(np.sum(np.abs(odom[1:, V]) * np.clip(dt_all, 0.0, MAX_DT_SEC)))
    print(f"Loaded {len(odom)} odom, {len(imu)} imu messages, {duration:.1f} s, "
          f"path length {path_len:.2f} m")

    # --- Sanity check 1: zero noise must reproduce clean odom's own pose.
    # If this fails, the integration or the wheel conversion is wrong and
    # nothing below means anything. It also confirms empirically that odom's
    # pose is the integral of its own twist (the Stage 4b argument).
    _, raw0 = corrupt_odometry(odom, args.b, 0.0, 0.0, 0.0, np.random.default_rng(0))
    e0 = errors(raw0, odom)
    print(f"[check] zero-noise raw vs truth: max pos {1000 * e0['pos'].max():.2f} mm, "
          f"max heading {np.degrees(np.abs(e0['heading']).max()):.3f} deg")

    # --- Sanity check 2: filter with CLEAN v should match truth (this is your
    # earlier live-validation result, reproduced offline).
    f0, _ = run_filter(odom, odom[:, V], imu)
    ef0 = errors(f0, odom)
    print(f"[check] filter with clean v vs truth: max pos {1000 * ef0['pos'].max():.2f} mm")

    # Predicted systematic heading drift of raw odometry: m * d / b (signed d)
    signed_d = float(np.sum(odom[1:, V] * np.clip(dt_all, 0.0, MAX_DT_SEC)))
    expected_drift_deg = np.degrees(args.mismatch * signed_d / args.b)
    print(f"Expected systematic raw heading drift ~ {expected_drift_deg:.1f} deg")
    # A constant gyro bias drifts the EKF heading by bias * total time.
    # It overtakes the wheels' drift at bias ~ wheel drift / duration.
    print(f"Predicted crossover gyro bias ~ {abs(expected_drift_deg) / duration:.4f} deg/s "
          f"(wheel drift / duration)")
    print()

    s_r, s_l = +args.mismatch / 2.0, -args.mismatch / 2.0

    if args.sweep:
        biases = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        raws = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            raws.append(corrupt_odometry(odom, args.b, s_r, s_l, args.sigma, rng,
                                         c=args.common_scale))
        er = [errors(r, odom) for _, r in raws]
        raw_h = np.mean([np.degrees(abs(e["heading"][-1])) for e in er])
        raw_p = np.mean([e["pos"][-1] for e in er])
        print(f"Gyro bias sweep, {args.seeds} seeds (raw odom does not depend on gyro bias)")
        print(f"{'bias [deg/s]':>13}{'raw heading':>13}{'EKF heading':>13}"
              f"{'raw pos [m]':>13}{'EKF pos [m]':>13}   EKF better?")
        for bias in biases:
            imu_b = imu.copy()
            imu_b[:, 1] += np.radians(bias)
            hs, ps = [], []
            for v_n, _ in raws:
                filt, _ = run_filter(odom, v_n, imu_b)
                e = errors(filt, odom)
                hs.append(np.degrees(abs(e["heading"][-1])))
                ps.append(e["pos"][-1])
            h, p_ = np.mean(hs), np.mean(ps)
            verdict = ("heading+pos" if (h < raw_h and p_ < raw_p)
                       else "pos only" if p_ < raw_p
                       else "heading only" if h < raw_h else "no")
            print(f"{bias:>13.3f}{raw_h:>13.2f}{h:>13.2f}{raw_p:>13.4f}{p_:>13.4f}   {verdict}")
        return

    # Apply the constant gyro bias for the normal (non-sweep) run
    imu = imu.copy()
    imu[:, 1] += np.radians(args.gyro_bias)

    keys = ["final |heading| [deg]", "final pos [m]", "RMS cross [m]", "RMS along [m]"]
    results = {"raw": {k: [] for k in keys}, "filtered": {k: [] for k in keys}}
    within = {"along": [], "cross": []}
    seed0 = None

    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        v_n, raw = corrupt_odometry(odom, args.b, s_r, s_l, args.sigma, rng, c=args.common_scale)
        filt, P_xy = run_filter(odom, v_n, imu)
        for name, track in (("raw", raw), ("filtered", filt)):
            e = errors(track, odom)
            results[name]["final |heading| [deg]"].append(np.degrees(abs(e["heading"][-1])))
            results[name]["final pos [m]"].append(e["pos"][-1])
            results[name]["RMS cross [m]"].append(rms(e["cross"]))
            results[name]["RMS along [m]"].append(rms(e["along"]))
        # Consistency: fraction of time the filter's error is inside its own
        # +-2 sigma bound. For a consistent filter this should be ~95%.
        # Only moving samples count; while stationary both error and sigma
        # are ~0 and would inflate the score.
        ef = errors(filt, odom)
        sa, sc = sigma_along_cross(P_xy, odom)
        moving = np.abs(odom[:, V]) > 1e-3
        within["along"].append(np.mean(np.abs(ef["along"][moving]) <= 2 * sa[moving]))
        within["cross"].append(np.mean(np.abs(ef["cross"][moving]) <= 2 * sc[moving]))
        if seed == 0:
            seed0 = (raw, filt, P_xy)

    print(f"{args.seeds} seeds, mismatch {100 * args.mismatch:.2f}%, sigma {100 * args.sigma:.1f}%, "
          f"common scale {100 * args.common_scale:.2f}%, gyro bias {args.gyro_bias:.3f} deg/s")
    print(f"{'metric':<24}{'raw (mean ± std)':>22}{'filtered (mean ± std)':>26}")
    for k in keys:
        r, f = np.array(results["raw"][k]), np.array(results["filtered"][k])
        print(f"{k:<24}{r.mean():>12.4f} ± {r.std():<8.4f}{f.mean():>15.4f} ± {f.std():<8.4f}")

    print()
    print("Filter consistency, fraction of moving samples inside +-2 sigma (target ~95%):")
    for k in ("along", "cross"):
        w = 100 * np.array(within[k])
        print(f"  {k:<6} {w.mean():6.1f}% (min over seeds {w.min():.1f}%)")

    if args.plot or args.save:
        import matplotlib
        if not args.plot:
            matplotlib.use("Agg")  # file output only, no window needed
        import matplotlib.pyplot as plt
        # Imported here, not at the top: compare_ground_truth imports this
        # module, so a top-level import would be circular. One colour per
        # track across all figures: clean odometry blue, noisy red, EKF green.
        from compare_ground_truth import track_style

        ref_lab, ref_col, _ = track_style("/odom")
        raw_lab, raw_col, _ = track_style("/odom_noisy")
        ekf_lab, ekf_col, _ = track_style("/odometry/filtered")
        ref_lab = f"{ref_lab} (reference)"

        raw, filt, P_xy = seed0
        t = odom[:, T] - odom[0, T]
        er, ef = errors(raw, odom), errors(filt, odom)
        sa, sc = sigma_along_cross(P_xy, odom)
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"Offline noise injection, seed 0 of {args.seeds}: "
                     f"mismatch {100 * args.mismatch:.1f}%, sigma {100 * args.sigma:.0f}% "
                     f"per wheel, gyro bias {args.gyro_bias:g} deg/s")

        ax[0, 0].plot(odom[:, PX], odom[:, PY], color=ref_col, linestyle="-", label=ref_lab)
        ax[0, 0].plot(raw[:, 0], raw[:, 1], color=raw_col, linestyle=":", label=raw_lab)
        ax[0, 0].plot(filt[:, 0], filt[:, 1], color=ekf_col, linestyle="-.", label=ekf_lab)
        ax[0, 0].set_aspect("equal")
        ax[0, 0].set_title("Trajectory")
        ax[0, 0].set_xlabel("x [m]")
        ax[0, 0].set_ylabel("y [m]")
        ax[0, 0].legend(fontsize="small")

        ax[0, 1].plot(t, np.degrees(er["heading"]), color=raw_col, label=raw_lab)
        ax[0, 1].plot(t, np.degrees(ef["heading"]), color=ekf_col, label=ekf_lab)
        ax[0, 1].set_xlabel("t [s]")
        ax[0, 1].set_ylabel("heading error [deg]")
        ax[0, 1].set_title("Heading error against clean odometry")
        ax[0, 1].legend(fontsize="small")

        for a, key, sig, name in ((ax[1, 0], "along", sa, "Along-track"),
                                  (ax[1, 1], "cross", sc, "Cross-track")):
            a.plot(t, 1000 * ef[key], color=ekf_col, label="EKF error")
            a.fill_between(t, -2000 * sig, 2000 * sig, color=ekf_col, alpha=0.15,
                           label="EKF ±2 sigma")
            a.set_xlabel("t [s]")
            a.set_ylabel("[mm]")
            a.set_title(f"{name}: EKF error vs its own claimed uncertainty")
            a.legend(fontsize="small")

        plt.tight_layout()
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"saved {args.save}")
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
