"""
Wheel-level odometry corruption for a differential-drive robot.

Live counterpart of corrupt_odometry() in test_noisy_odom.py. Same noise
model, same wheel split, same Euler step, same dt guard, and the same
operation order, so for identical inputs and noise draws the two produce
bit-identical results (Check A, check_wheel_noise.py).

Noise model, per wheel, per odometry message:
    v_wheel_noisy = v_wheel * (1 + c) * (1 + s) * (1 + n)
    c  common scale error on both wheels (wrong nominal wheel radius)
    s  differential scale error, s_r = +m/2, s_l = -m/2
       (wheel diameter mismatch, Borenstein & Feng 1996)
    n  white multiplicative noise, one draw per wheel per message

The noise samples n are passed in, not drawn here. That keeps the class
deterministic, so it can be compared against the offline script with the
exact same draws. The ROS node owns the RNG.

Integration, per message k (matches corrupt_odometry exactly):
    dt = t_k - t_{k-1}                      (from header.stamp)
    if 0 < dt <= max_dt:
        x     += v_n[k] * cos(theta_{k-1}) * dt
        y     += v_n[k] * sin(theta_{k-1}) * dt
        theta  = wrap(theta_{k-1} + w_n[k] * dt)
    else:
        hold the pose (the stamp is still updated)

The current message's twist is applied over the preceding interval with
the previous heading. The pose is the integral of the published twist, as
on a real robot, so pose and twist stay consistent.

No ROS dependency on purpose. Run this file for the self-test.
"""

import numpy as np

from tb3_state_estimation.differential_drive_ekf import wrap_angle

# Same guard as ekf_node.py and test_noisy_odom.py (5 x the 50 Hz period).
DEFAULT_MAX_DT_SEC = 5.0 * (1.0 / 50.0)


class WheelOdomCorruptor:
    """Corrupts odometry at the wheel level and integrates its own pose."""

    def __init__(
        self,
        wheel_separation: float = 0.160,
        mismatch: float = 0.005,
        common_scale: float = 0.0,
        max_dt: float = DEFAULT_MAX_DT_SEC,
    ):
        if wheel_separation <= 0.0:
            raise ValueError("wheel_separation must be positive")
        if max_dt <= 0.0:
            raise ValueError("max_dt must be positive")
        self.b = float(wheel_separation)
        self.c = float(common_scale)
        self.s_r = +float(mismatch) / 2.0
        self.s_l = -float(mismatch) / 2.0
        self.max_dt = float(max_dt)

        self.last_t = None
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    @property
    def initialized(self) -> bool:
        return self.last_t is not None

    @property
    def pose(self):
        return self.x, self.y, self.theta

    def initialize(self, t: float, x: float, y: float, theta: float) -> None:
        """Start from the first clean message's pose, like track[0] offline."""
        self.last_t = float(t)
        self.x, self.y, self.theta = float(x), float(y), float(theta)

    def corrupt_twist(self, v: float, w: float, n_r: float, n_l: float):
        """Return (v_noisy, w_noisy). Pure function of the inputs."""
        b = self.b
        # Operation order mirrors corrupt_odometry() so floats match exactly.
        v_r = v + w * b / 2.0
        v_l = v - w * b / 2.0
        v_r_n = v_r * (1.0 + self.c) * (1.0 + self.s_r) * (1.0 + n_r)
        v_l_n = v_l * (1.0 + self.c) * (1.0 + self.s_l) * (1.0 + n_l)
        return (v_r_n + v_l_n) / 2.0, (v_r_n - v_l_n) / b

    def step(self, t: float, v: float, w: float, n_r: float, n_l: float):
        """Process one odometry message after the first.

        Returns (v_noisy, w_noisy, dt). The caller can check dt against
        the guard to log skipped integration steps."""
        if not self.initialized:
            raise RuntimeError("call initialize() with the first message first")

        v_n, w_n = self.corrupt_twist(v, w, n_r, n_l)
        dt = t - self.last_t
        self.last_t = t

        if 0.0 < dt <= self.max_dt:
            # theta is updated last: x and y use the previous heading.
            self.x += v_n * np.cos(self.theta) * dt
            self.y += v_n * np.sin(self.theta) * dt
            self.theta = wrap_angle(self.theta + w_n * dt)

        return v_n, w_n, dt


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    dt = 0.02

    # 1) Zero noise: noisy twist equals input twist (only round-off).
    z = WheelOdomCorruptor(mismatch=0.0, common_scale=0.0)
    worst = 0.0
    for v, w in rng.uniform(-0.3, 0.3, size=(1000, 2)):
        v_n, w_n = z.corrupt_twist(v, w, 0.0, 0.0)
        worst = max(worst, abs(v_n - v), abs(w_n - w))
    assert worst < 1e-12, worst
    print(f"Zero-noise twist round trip: OK (max diff {worst:.1e})")

    # 2) Straight line with mismatch m: heading drift must equal m * d / b.
    m, b, v, N = 0.005, 0.160, 0.2, 500
    s = WheelOdomCorruptor(wheel_separation=b, mismatch=m)
    s.initialize(0.0, 0.0, 0.0, 0.0)
    for k in range(1, N):
        s.step(k * dt, v, 0.0, 0.0, 0.0)
    d = v * dt * (N - 1)
    expected = m * d / b
    assert abs(s.theta - expected) < 1e-12, (s.theta, expected)
    print(f"Straight-line drift m*d/b: OK ({np.degrees(s.theta):.4f} deg over {d:.3f} m)")

    # 3) Common scale c on a straight line: distance scales by (1 + c).
    c = 0.01
    cs = WheelOdomCorruptor(mismatch=0.0, common_scale=c)
    cs.initialize(0.0, 0.0, 0.0, 0.0)
    for k in range(1, N):
        cs.step(k * dt, v, 0.0, 0.0, 0.0)
    assert abs(cs.x - d * (1.0 + c)) < 1e-12 and abs(cs.y) < 1e-15, cs.pose
    print(f"Common scale: OK (x = {cs.x:.6f} m, expected {d * (1 + c):.6f} m)")

    # 4) Stationary with noise on: multiplicative noise on zero speed is zero.
    st = WheelOdomCorruptor(mismatch=0.005, common_scale=0.01)
    st.initialize(0.0, 1.0, 2.0, 0.5)
    for k in range(1, 200):
        st.step(k * dt, 0.0, 0.0, *rng.normal(0.0, 0.03, 2))
    assert st.pose == (1.0, 2.0, 0.5), st.pose
    print("Stationary with noise: OK (pose frozen)")

    # 5) dt guards: a large gap and a backwards stamp both hold the pose,
    #    and integration resumes normally from the new stamp.
    g = WheelOdomCorruptor(mismatch=0.0)
    g.initialize(0.0, 0.0, 0.0, 0.0)
    g.step(0.5, v, 0.0, 0.0, 0.0)     # gap 0.5 s > max_dt: skipped
    g.step(0.4, v, 0.0, 0.0, 0.0)     # backwards: skipped
    assert g.pose == (0.0, 0.0, 0.0), g.pose
    g.step(0.42, v, 0.0, 0.0, 0.0)    # normal 0.02 s step
    assert abs(g.x - v * 0.02) < 1e-15, g.pose
    print("dt guards: OK (gap and backwards stamp skipped, then resumes)")

    print("All wheel_noise self-tests passed.")
