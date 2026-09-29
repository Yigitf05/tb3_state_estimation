#!/usr/bin/env python3
"""
Stage 4 live, steps 2 and 3: clean /odom, /odom_noisy and the live EKF's
/odometry/filtered against Gazebo ground truth.

Ground truth comes from the OdometryPublisher system added to the burger's
model.sdf, bridged to ROS as /ground_truth (nav_msgs/Odometry, frame "world").

Frames. /odom, /odom_noisy and /odometry/filtered live in their own odom
frames, whose origin is wherever each one started integrating. Ground truth
lives in Gazebo's world frame. They are related by one fixed SE(2) transform
per track, fixed by anchoring at a stationary instant t0 (0.5 s before the
robot first moves, inside the first stationary stretch):

    T_WO     = G(t0) o O(t0)^-1        world <- odom
    G_O(t)   = T_WO^-1 o G(t)          truth expressed in that odom frame

Anchoring at the start, NOT a least-squares fit over the whole run: a fitted
rotation would absorb exactly the heading drift we want to measure
(Zhang & Scaramuzza, IROS 2018). Each track is anchored separately, so the
errors are those accumulated during the recording. A consequence: a constant
offset a track already had at t0 (for example the EKF's stationary heading
random walk before motion) is removed. Initialization is checked separately
(check_live_ekf.py, E1a).

Times come from header.stamp (load_bag), never the bag receive time, so no
use_sim_time is involved anywhere. /odometry/filtered copies the stamp of the
/odom_noisy message that triggered it, which copies /odom's, so all tracks
share stamps with /ground_truth.

Decision rule, fixed before the step 2 run (clean /odom vs truth, after t0):
    max pos <= 5 mm and max |heading| <= 0.05 deg   -> offline results stand
    max pos <= 50 mm and max |heading| <= 0.1 deg   -> raw headline numbers
        stand; EKF mm-level numbers need truth from ground truth
    otherwise                                       -> diagnose first

Step 3 prediction (live EKF vs truth): max heading error below 0.05 deg, and
position error following clean /odom's own turn-driven curve, within ~20 mm.

Usage:
    python3 compare_ground_truth.py <bag_dir> [--until <s after t0>] [--plot]
    python3 compare_ground_truth.py --self-test
"""

import argparse

import numpy as np

from tb3_state_estimation.differential_drive_ekf import wrap_angle
from test_noisy_odom import PX, PY, T, V, W, YAW, errors, load_bag, rms

STAMP_TOL_SEC = 1e-6     # "same stamp" for exact-match counting
V_STILL = 1e-3           # [m/s]   below this, odom twist counts as stationary
W_STILL = 1e-3           # [rad/s]
MIN_STILL_SEC = 2.0      # shorter stationary stretches are ignored
ANCHOR_LEAD_SEC = 0.5    # anchor this long before the first motion
SPIN_V_MAX = 0.01        # [m/s]   in-place spin: tiny v ...
SPIN_W_MIN = 0.2         # [rad/s] ... and a real turn rate


# ---------------------------------------------------------------- SE(2) ---

def compose(a, b):
    """a o b for poses [x, y, yaw]; either argument may be (3,) or (N, 3)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([
        a[..., 0] + c * b[..., 0] - s * b[..., 1],
        a[..., 1] + s * b[..., 0] + c * b[..., 1],
        wrap_angle(a[..., 2] + b[..., 2]),
    ], axis=-1)


def inverse(a):
    a = np.asarray(a, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([
        -(c * a[..., 0] + s * a[..., 1]),
        -(-s * a[..., 0] + c * a[..., 1]),
        -a[..., 2],
    ], axis=-1)


# --------------------------------------------------------- time matching ---

def truth_at(track, truth):
    """Ground-truth pose [x, y, yaw] at each track stamp.

    Linear interpolation, yaw on the unwrapped angle. At an exact stamp match
    interpolation returns the sample itself, so this is exact whenever the
    stamps line up. Returns (poses, keep_mask, n_exact)."""
    tt, tk = truth[:, T], track[:, T]
    keep = (tk >= tt[0]) & (tk <= tt[-1])
    tk = tk[keep]
    yaw_u = np.unwrap(truth[:, YAW])
    poses = np.stack([
        np.interp(tk, tt, truth[:, PX]),
        np.interp(tk, tt, truth[:, PY]),
        wrap_angle(np.interp(tk, tt, yaw_u)),
    ], axis=-1)
    j = np.clip(np.searchsorted(tt, tk), 1, len(tt) - 1)
    nearest = np.minimum(np.abs(tt[j] - tk), np.abs(tt[j - 1] - tk))
    return poses, keep, int(np.sum(nearest <= STAMP_TOL_SEC))


# ------------------------------------------------------------- segments ---

def runs(mask):
    """(start, end) index pairs of consecutive True runs, end inclusive."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def still_segments(odom):
    mask = (np.abs(odom[:, V]) < V_STILL) & (np.abs(odom[:, W]) < W_STILL)
    return [(i, j) for i, j in runs(mask)
            if odom[j, T] - odom[i, T] >= MIN_STILL_SEC]


def anchor_time(odom):
    """t0: 0.5 s before the first motion, inside the first stationary stretch.

    Not the last stationary sample: that sample's truth neighbours can
    straddle the first moving sample, and any interpolation or one-physics-
    step offset there would become a permanent offset in every error."""
    stills = still_segments(odom)
    if not stills:
        raise SystemExit("No stationary stretch >= %.0f s in /odom; cannot anchor."
                         % MIN_STILL_SEC)
    i0, i1 = stills[0]
    return max(odom[i0, T], odom[i1, T] - ANCHOR_LEAD_SEC)


# ------------------------------------------------------------- analysis ---

def analyze_track(name, track, truth, anchor_t):
    """Align truth into this track's odom frame at anchor_t and return
    everything the report needs. `track` is a load_bag-style (N, 6) array."""
    G, keep, n_exact = truth_at(track, truth)
    tr = track[keep]
    k0 = int(np.argmin(np.abs(tr[:, T] - anchor_t)))
    O = tr[:, [PX, PY, YAW]]

    T_WO = compose(G[k0], inverse(O[k0]))
    G_O = compose(inverse(T_WO), G)

    # From the anchor on only. Truth goes into the PX/PY/YAW columns so the
    # existing errors() from test_noisy_odom.py applies unchanged.
    O, G_O, tr = O[k0:], G_O[k0:], tr[k0:]
    ref = tr.copy()
    ref[:, [PX, PY, YAW]] = G_O
    e = errors(O, ref)

    # Scale fit: position error ~ c * displacement from the anchor.
    d = G_O[:, :2] - G_O[0, :2]
    err = O[:, :2] - G_O[:, :2]
    dd = float(np.sum(d * d))
    c = float(np.sum(err * d) / dd) if dd > 0 else float("nan")
    resid = err - c * d
    resid_rms = rms(np.hypot(resid[:, 0], resid[:, 1]))

    # Rotation scale over in-place spins (the track's own twist decides what
    # a spin is; for the EKF that is odometry v and the filtered omega).
    spin = (np.abs(tr[:, V]) < SPIN_V_MAX) & (np.abs(tr[:, W]) > SPIN_W_MIN)
    step = spin[1:]
    d_odom = np.diff(np.unwrap(O[:, 2]))[step].sum()
    d_true = np.diff(np.unwrap(G_O[:, 2]))[step].sum()

    return {
        "name": name, "n": len(tr), "n_track": len(track), "n_exact": n_exact,
        "t": tr[:, T], "O": O, "G_O": G_O, "e": e, "T_WO": T_WO,
        "c": c, "c_resid_rms": resid_rms,
        "spin_odom": d_odom, "spin_true": d_true,
    }


def verdict(r):
    p = 1000 * r["e"]["pos"].max()
    h = np.degrees(np.abs(r["e"]["heading"]).max())
    if p <= 5.0 and h <= 0.05:
        return p, h, "OFFLINE RESULTS STAND (clean /odom is a valid truth)"
    if p <= 50.0 and h <= 0.1:
        return p, h, ("RAW HEADLINE NUMBERS STAND; EKF mm-level numbers need "
                      "ground truth as the reference")
    return p, h, "OUTSIDE BOUNDS: diagnose before touching offline results"


def report(odom, truth, extras=(), quiet=False, until=None):
    """Compare clean /odom and any extra tracks, given as (name, track)
    pairs, against ground truth. All tracks share the anchor t0."""
    anchor_t = anchor_time(odom)
    extras = list(extras)

    # Optional analysis window: keep only data up to `until` seconds after
    # t0 (e.g. to judge the run before a collision). The anchor is unchanged.
    if until is not None:
        odom = odom[odom[:, T] <= anchor_t + until]
        extras = [(n, tr[tr[:, T] <= anchor_t + until]) for n, tr in extras]
    stills = still_segments(odom)

    results = [analyze_track("/odom", odom, truth, anchor_t)]
    results += [analyze_track(n, tr, truth, anchor_t) for n, tr in extras]
    if quiet:
        return results, stills, anchor_t

    t0 = odom[0, T]
    dt_all = np.clip(np.diff(odom[:, T]), 0.0, 0.25)
    path = float(np.sum(np.abs(odom[1:, V]) * dt_all))
    print(f"/odom {len(odom)} msgs, /ground_truth {len(truth)} msgs, "
          f"{odom[-1, T] - t0:.1f} s, path {path:.2f} m"
          + ("" if until is None else f"  [window: t - t0 <= {until:.1f} s]"))
    print(f"anchor t0 = {anchor_t - t0:.2f} s after first /odom stamp "
          f"({ANCHOR_LEAD_SEC} s before first motion)")
    tw = results[0]["T_WO"]
    print(f"world <- odom (clean): x {tw[0]:+.4f} m, y {tw[1]:+.4f} m, "
          f"yaw {np.degrees(tw[2]):+.3f} deg")

    print("\nStationary stretches (ground-truth motion that odom cannot see):")
    print(f"  {'start [s]':>9} {'dur [s]':>8} {'truth moved [mm]':>17} "
          f"{'odom moved [mm]':>16} {'truth dyaw [deg]':>17}")
    for i, j in stills:
        g, keep, _ = truth_at(odom[[i, j]], truth)
        if keep.sum() < 2:
            continue
        o = odom[[i, j]][:, [PX, PY]]
        print(f"  {odom[i, T] - t0:>9.1f} {odom[j, T] - odom[i, T]:>8.1f} "
              f"{1000 * np.hypot(*(g[1, :2] - g[0, :2])):>17.3f} "
              f"{1000 * np.hypot(*(o[1] - o[0])):>16.3f} "
              f"{np.degrees(wrap_angle(g[1, 2] - g[0, 2])):>17.4f}")

    for r in results:
        e = r["e"]
        print(f"\n== {r['name']} vs ground truth (from t0) ==")
        print(f"  stamps: {r['n_exact']}/{r['n_track']} exact matches with /ground_truth"
              + ("" if r["n_exact"] == r["n_track"] else "  (rest interpolated)"))
        print(f"  heading  final {np.degrees(e['heading'][-1]):+.4f} deg, "
              f"max |.| {np.degrees(np.abs(e['heading']).max()):.4f} deg")
        print(f"  position final {1000 * e['pos'][-1]:.2f} mm, "
              f"max {1000 * e['pos'].max():.2f} mm")
        # Where the error first breaks out: locates events such as a collision.
        for thr_mm in (5, 20, 100):
            hit = np.flatnonzero(1000 * e["pos"] > thr_mm)
            when = (f"t - t0 = {r['t'][hit[0]] - anchor_t:.1f} s" if len(hit)
                    else "never")
            print(f"    first exceeds {thr_mm:>3} mm: {when}")
        print(f"  along    final {1000 * e['along'][-1]:+.2f} mm, "
              f"RMS {1000 * rms(e['along']):.2f} mm")
        print(f"  cross    final {1000 * e['cross'][-1]:+.2f} mm, "
              f"RMS {1000 * rms(e['cross']):.2f} mm")
        print(f"  scale fit: error ~ c * displacement, c = {100 * r['c']:+.4f} %, "
              f"residual RMS {1000 * r['c_resid_rms']:.2f} mm"
              "  (a scale error only if the residual is small)")
        if abs(r["spin_true"]) > 2 * np.pi:
            print(f"  in-place spins: {r['name']} {np.degrees(r['spin_odom']):.1f} deg / "
                  f"truth {np.degrees(r['spin_true']):.1f} deg = "
                  f"ratio {r['spin_odom'] / r['spin_true']:.5f}")
        else:
            print("  in-place spins: < 1 full turn recorded, ratio not reported")

    p, h, v = verdict(results[0])
    print(f"\nDecision rule (clean /odom): max pos {p:.2f} mm, "
          f"max heading {h:.4f} deg -> {v}")
    return results, stills, anchor_t


# ------------------------------------------------------------ self-test ---

def simulate(t, v_cmd, w_cmd, start, c=0.0, r=1.0):
    """Exact unicycle integration, command k held over [t[k-1], t[k]].
    Returns a load_bag-style (N, 6) array; c scales v, r scales omega."""
    out = np.zeros((len(t), 6))
    x, y, th = start
    out[0] = (t[0], v_cmd[0], w_cmd[0], x, y, th)
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        v, w = v_cmd[k] * (1 + c), w_cmd[k] * r
        if abs(w) < 1e-12:
            x += v * dt * np.cos(th)
            y += v * dt * np.sin(th)
        else:
            x += v / w * (np.sin(th + w * dt) - np.sin(th))
            y -= v / w * (np.cos(th + w * dt) - np.cos(th))
        th = wrap_angle(th + w * dt)
        out[k] = (t[k], v, w, x, y, th)
    return out


def self_test():
    rate = 50.0
    plan = [  # (duration s, v m/s, w rad/s)
        (10, 0, 0), (10, 0.2, 0), (4 * np.pi / 1.0, 0, 1.0), (15, 0.15, 0.3),
        (5, 0, 0), (8, -0.1, 0), (12, 0.22, -0.2), (10, 0, 0),
    ]
    v_cmd, w_cmd = [0.0], [0.0]
    for dur, v, w in plan:
        n = int(round(dur * rate))
        v_cmd += [v] * n
        w_cmd += [w] * n
    t = 100.0 + np.arange(len(v_cmd)) / rate  # sim-time stamps
    v_cmd, w_cmd = np.array(v_cmd), np.array(w_cmd)
    spawn = (-2.0, -0.5, np.radians(30.0))    # rotated spawn on purpose

    truth = simulate(t, v_cmd, w_cmd, spawn)
    ok = True

    def check(label, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}: {detail}")

    print("Self-test 1: perfect odometry, spawn yaw 30 deg (tests alignment)")
    odom = simulate(t, v_cmd, w_cmd, (0.0, 0.0, 0.0))
    (r,), _, _ = report(odom, truth, quiet=True)
    check("zero error after SE(2) anchoring", r["e"]["pos"].max() < 1e-9,
          f"max pos {r['e']['pos'].max():.1e} m")
    naive = np.hypot(odom[:, PX] - (truth[:, PX] - truth[0, PX]),
                     odom[:, PY] - (truth[:, PY] - truth[0, PY])).max()
    check("naive position subtraction would be wrong", naive > 0.1,
          f"naive max error {naive:.3f} m")
    check("all stamps matched exactly", r["n_exact"] == len(odom),
          f"{r['n_exact']}/{len(odom)}")

    print("Self-test 2: 0.3 % common scale error (tests the scale fit)")
    odom = simulate(t, v_cmd, w_cmd, (0.0, 0.0, 0.0), c=0.003)
    (r,), _, _ = report(odom, truth, quiet=True)
    check("fitted c recovers 0.3 %", abs(r["c"] - 0.003) < 1e-5,
          f"c = {100 * r['c']:.4f} %")
    check("heading unaffected", np.abs(r["e"]["heading"]).max() < 1e-9,
          f"max {np.degrees(np.abs(r['e']['heading']).max()):.1e} deg")

    print("Self-test 3: 0.1 % rotation scale error, odom started earlier (tests "
          "spin ratio and per-track anchoring)")
    odom = simulate(t, v_cmd, w_cmd, (1.0, 2.0, -1.0), r=1.001)
    (r,), _, _ = report(odom, truth, quiet=True)
    ratio = r["spin_odom"] / r["spin_true"]
    check("spin ratio recovers 1.001", abs(ratio - 1.001) < 1e-6, f"{ratio:.6f}")
    check("heading error at t0 is zero", abs(r["e"]["heading"][0]) < 1e-12,
          f"{r['e']['heading'][0]:.1e} rad")

    print("Self-test 4: truth at 50 Hz offset by half a period (tests interpolation)")
    # Same physical trajectory, sampled exactly at mid-period instants:
    # propagate each 50 Hz truth pose forward 10 ms under the command that
    # holds over that interval.
    truth_off = np.zeros((len(t) - 1, 6))
    for k in range(1, len(t)):
        seg = simulate(np.array([t[k - 1], t[k - 1] + 0.01]),
                       v_cmd[[k, k]], w_cmd[[k, k]], truth[k - 1, [PX, PY, YAW]])
        truth_off[k - 1] = seg[1]
    odom = simulate(t, v_cmd, w_cmd, (0.0, 0.0, 0.0))
    (r,), _, _ = report(odom, truth_off, quiet=True)
    # The synthetic commands jump instantly, so the two samples bracketing a
    # step straddle two different velocities and linear interpolation is off
    # by ~dv * 5 ms there (1-2 mm here). Gazebo's DiffDrive is acceleration
    # limited to 1 m/s^2, i.e. dv <= 0.02 m/s per period, so ~0.1 mm at most.
    # Judge the interpolation away from the steps.
    tt = r["t"]
    change = np.flatnonzero((np.diff(v_cmd) != 0) | (np.diff(w_cmd) != 0))
    step_t = t[change]
    near = np.any(np.abs(tt[:, None] - step_t[None, :]) < 0.05, axis=1)
    away = 1000 * r["e"]["pos"][~near].max()
    check("interpolated comparison away from command steps < 0.01 mm", away < 0.01,
          f"{away:.5f} mm (at steps {1000 * r['e']['pos'][near].max():.2f} mm), "
          f"exact matches {r['n_exact']}")

    print("Self-test 5: extra track identical to /odom (tests the extras path)")
    odom = simulate(t, v_cmd, w_cmd, (0.0, 0.0, 0.0))
    (r0, r1), _, _ = report(odom, truth, [("copy", odom.copy())], quiet=True)
    same = np.max(np.abs(r0["e"]["pos"] - r1["e"]["pos"]))
    check("extra track gives identical errors", same == 0.0, f"max diff {same:.1e} m")

    print("\nSELF-TEST " + ("PASSED" if ok else "FAILED"))
    return ok


# ----------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", nargs="?")
    ap.add_argument("--truth-topic", default="/ground_truth")
    ap.add_argument("--noisy-topic", default="/odom_noisy")
    ap.add_argument("--filtered-topic", default="/odometry/filtered")
    ap.add_argument("--until", type=float, default=None,
                    help="analyse only up to this many seconds after t0")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not args.bag:
        ap.error("bag directory required (or --self-test)")

    odom, _ = load_bag(args.bag, odom_topic="/odom")
    truth, _ = load_bag(args.bag, odom_topic=args.truth_topic)
    extras = []
    for topic in (args.noisy_topic, args.filtered_topic):
        try:
            track, _ = load_bag(args.bag, odom_topic=topic)
            extras.append((topic, track))
        except SystemExit as exc:
            print(f"(no {topic}: {exc})")

    results, _, anchor_t = report(odom, truth, extras, until=args.until)

    for r in results:
        if r["name"] == args.filtered_topic:
            h = np.degrees(np.abs(r["e"]["heading"]).max())
            p = 1000 * r["e"]["pos"].max()
            p_odom = 1000 * results[0]["e"]["pos"].max()
            print(f"Step 3 prediction ({r['name']} vs truth): max heading {h:.4f} deg "
                  f"(predicted < 0.05); max pos {p:.2f} mm against clean /odom's "
                  f"{p_odom:.2f} mm (predicted: same curve, within ~20 mm)")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        r0 = results[0]
        ax[0, 0].plot(r0["G_O"][:, 0], r0["G_O"][:, 1], "k-", label="ground truth")
        styles = ("b--", "r:", "g-")
        colors = ("b", "r", "g")
        for r, style in zip(results, styles):
            ax[0, 0].plot(r["O"][:, 0], r["O"][:, 1], style, label=r["name"])
        ax[0, 0].set_aspect("equal")
        ax[0, 0].set_title("Trajectories in the clean odom frame")
        ax[0, 0].legend()
        for r, col in zip(results, colors):
            tt = r["t"] - anchor_t
            ax[0, 1].plot(tt, np.degrees(r["e"]["heading"]), col, label=r["name"])
            ax[1, 0].plot(tt, 1000 * r["e"]["along"], col, label=r["name"])
            ax[1, 1].plot(tt, 1000 * r["e"]["cross"], col, label=r["name"])
        for a, lab in ((ax[0, 1], "heading error [deg]"),
                       (ax[1, 0], "along-track error [mm]"),
                       (ax[1, 1], "cross-track error [mm]")):
            a.set_xlabel("t - t0 [s]")
            a.set_ylabel(lab)
            a.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
