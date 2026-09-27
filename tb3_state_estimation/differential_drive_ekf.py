"""
Extended Kalman Filter for 2D differential-drive robot pose estimation.

State vector:   x = [x, y, theta, omega]^T
    x, y      - position in the odom frame (meters)
    theta     - heading (radians, wrapped to [-pi, pi])
    omega     - angular velocity (rad/s), estimated as its own state so the
                IMU gyro measurement has something valid to correct against.
                A gyro measures a rate, not an angle, so it cannot directly
                correct theta -- it corrects omega, which then feeds theta
                on the next predict step.

Process model (nonlinear -- this nonlinearity is why it's an "Extended" KF,
not a plain linear KF):
    x_k     = x_{k-1} + v * cos(theta_{k-1}) * dt
    y_k     = y_{k-1} + v * sin(theta_{k-1}) * dt
    theta_k = theta_{k-1} + omega_{k-1} * dt
    omega_k = omega_{k-1}                          (constant-turn-rate assumption
                                                       between predict ticks;
                                                       Q covers the drift)

    v (linear velocity) is treated as a trusted control input taken from
    wheel odometry (/odom twist.linear.x), not a filtered state -- Gazebo's
    simulated odometry currently reports ~zero noise, so there's nothing
    to gain from filtering it yet. See project notes for the plan to
    revisit this once synthetic wheel-slip noise is injected.

Measurement model (update step):
    z = omega_z, the /imu angular_velocity.z reading (gyro yaw rate)
    H = [0, 0, 0, 1]   -- the gyro observes omega directly, nothing else
    R = measured from Gazebo's own imu angular_velocity_covariance (real
        number pulled from the actual /imu topic on this setup, not guessed)

This class has no ROS 2 dependency on purpose. It's meant to be validated
standalone first (see the __main__ block below, which runs it against a
synthetic trajectory with known ground truth) before any rclpy wiring
touches it.

References:
    Welch & Bishop, "An Introduction to the Kalman Filter", UNC TR 95-041
        https://www.cs.utexas.edu/~pstone/Courses/393Rfall15/readings/Welch+Bishop-TR-95.pdf
    Thrun, Burgard, Fox, "Probabilistic Robotics", ch. 3 (EKF) and ch. 5
        (velocity motion model, the v*dt / omega*dt process-noise scaling
        used below)
        http://ais.informatik.uni-freiburg.de/teaching/ws17/mapping/pdf/slam04-ekf.pdf
"""

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def yaw_to_quaternion(yaw: float):
    """Quaternion (x, y, z, w) for a pure yaw rotation."""
    return (0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0)))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw from a quaternion. The general form stays correct even if the
    source carries small roll or pitch. Returns a value in [-pi, pi]."""
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

class DifferentialDriveEKF:
    """EKF estimating [x, y, theta, omega] for a differential-drive robot,
    fusing wheel odometry (as a control input) with IMU gyro yaw rate
    (as a measurement).
    """

    def __init__(
        self,
        initial_state=None,
        initial_covariance=None,
        v_noise_ratio: float = 0.021,
        omega_accel_std: float = 1.0,
        measurement_noise: float = 3.999999975690116e-08,
    ):
        # State: [x, y, theta, omega]
        self.x = np.zeros(4) if initial_state is None else np.array(initial_state, dtype=float)

        # State covariance P (4x4) - start with modest uncertainty on everything
        self.P = np.eye(4) * 0.1 if initial_covariance is None else np.array(initial_covariance, dtype=float)

        # Process noise, derived from physical sources (see predict()):
        #   v_noise_ratio    sigma_v / |v|: std of odometry speed error per
        #                    message, relative to speed. 0.021 = 3% per-wheel
        #                    white noise / sqrt(2), matching the injected model.
        #   omega_accel_std  sqrt of the angular-acceleration power spectral
        #                    density driving the omega random walk,
        #                    in rad/s^2/sqrt(Hz).
        self.v_noise_ratio = float(v_noise_ratio)
        self.omega_accel_std = float(omega_accel_std)

        # Measurement noise R for the gyro (scalar - we only measure omega).
        # Default is the real variance read off Gazebo's
        # /imu angular_velocity_covariance[8] on this setup, not a guess.
        self.R = float(measurement_noise)

        # Measurement matrix: gyro observes omega (state index 3) only
        self.H = np.array([[0.0, 0.0, 0.0, 1.0]])

    def predict(self, v: float, dt: float) -> None:
        """Propagate the state forward using odometry's linear velocity v
        (m/s) as a trusted control input, over a timestep dt (seconds)."""
        x, y, theta, omega = self.x

        # Nonlinear motion model g(x, u)
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = wrap_angle(theta + omega * dt)
        omega_new = omega  # constant-turn-rate assumption; Q covers drift

        self.x = np.array([x_new, y_new, theta_new, omega_new])

        # Jacobian F = d g / d x, evaluated at the PREVIOUS state (the
        # linearization point every EKF predict step requires)
        F = np.array(
            [
                [1.0, 0.0, -v * np.sin(theta) * dt, 0.0],
                [0.0, 1.0, v * np.cos(theta) * dt, 0.0],
                [0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Process noise from two physical sources:
        # 1) Odometry speed error, mapped through V = dg/dv. Puts noise only
        #    along the heading direction and is zero when v = 0.
        V = np.array([np.cos(theta) * dt, np.sin(theta) * dt, 0.0, 0.0])
        sigma_v = self.v_noise_ratio * abs(v)
        Q = np.outer(V, V) * sigma_v**2
        # 2) Constant-turn-rate model error: omega is a random walk driven by
        #    angular acceleration. No direct theta term, theta inherits it
        #    through F[2,3] = dt. (The exact discretization adds tiny
        #    theta-omega terms of order dt^2, dt^3; negligible at dt = 0.02.)
        
        Q[3, 3] += self.omega_accel_std**2 * dt

        self.P = F @ self.P @ F.T + Q

    def update_gyro(self, omega_measured: float, R: float = None) -> None:
        """Correct the omega state (and, through the covariance, theta's
        future predictions) using a gyro yaw-rate measurement."""
        R = self.R if R is None else R

        z = np.array([omega_measured])
        z_pred = self.H @ self.x

        innovation = z - z_pred
        S = self.H @ self.P @ self.H.T + R           # innovation covariance (1x1)
        K = (self.P @ self.H.T) / S                    # Kalman gain, Nx1
        

        self.x = self.x + (K.flatten() * innovation[0])
        self.x[2] = wrap_angle(self.x[2])

        # Joseph form: algebraically equal to (I - KH)P for the optimal K,
        # but stays symmetric and positive semi-definite under rounding.
        IKH = np.eye(4) - K @ self.H          # (4x1) @ (1x4) -> 4x4
        self.P = IKH @ self.P @ IKH.T + R * (K @ K.T)
        

    @property
    def pose(self):
        """Return (x, y, theta) - the part you'll actually publish."""
        return self.x[0], self.x[1], self.x[2]


if __name__ == "__main__":

    for yaw in np.linspace(-np.pi + 1e-6, np.pi - 1e-6, 37):
        back = quaternion_to_yaw(*yaw_to_quaternion(yaw))
        assert abs(wrap_angle(back - yaw)) < 1e-9, (yaw, back)
    print("Quaternion round-trip: OK")
    # Standalone validation against synthetic ground truth, per the plan:
    # prove the math is right before any ROS 2 node touches it.
    #
    # Scenario: robot drives a circle at constant v and true omega. We know
    # the exact ground-truth trajectory analytically. We simulate a noisy
    # gyro (using the REAL measured noise std from /imu) and check that the
    # filter's omega/theta estimate tracks ground truth better than just
    # taking the raw noisy gyro reading at face value.

    rng = np.random.default_rng(seed=42)

    dt = 0.05          # 20 Hz, matches a typical /imu publish rate
    steps = 400         # 20 seconds
    true_v = 0.2        # m/s
    true_omega = 0.3    # rad/s, constant turn rate
    gyro_std = np.sqrt(3.999999975690116e-08)  # real measured IMU gyro noise

    ekf = DifferentialDriveEKF(initial_state=[0.0, 0.0, 0.0, 0.0])

    true_theta = 0.0
    raw_integrated_theta = 0.0  # naive: integrate the noisy gyro directly, no filter

    filtered_theta_err = []
    raw_theta_err = []
    filtered_omega_err = []

    for _ in range(steps):
        # ground truth advances exactly
        true_theta = wrap_angle(true_theta + true_omega * dt)

        # simulated noisy gyro measurement
        gyro_meas = true_omega + rng.normal(0.0, gyro_std)

        # naive baseline: just integrate the raw noisy gyro, no filtering
        raw_integrated_theta = wrap_angle(raw_integrated_theta + gyro_meas * dt)

        # EKF: predict using trusted v, correct using noisy gyro
        ekf.predict(v=true_v, dt=dt)
        ekf.update_gyro(gyro_meas)

        filtered_theta_err.append(abs(wrap_angle(ekf.x[2] - true_theta)))
        raw_theta_err.append(abs(wrap_angle(raw_integrated_theta - true_theta)))
        filtered_omega_err.append(abs(ekf.x[3] - true_omega))

    filtered_theta_rmse = np.sqrt(np.mean(np.square(filtered_theta_err)))
    raw_theta_rmse = np.sqrt(np.mean(np.square(raw_theta_err)))
    filtered_omega_rmse = np.sqrt(np.mean(np.square(filtered_omega_err)))

    print(f"Steps simulated:                 {steps} ({steps*dt:.1f} s)")
    print(f"True v, true omega:              {true_v} m/s, {true_omega} rad/s")
    print(f"Gyro noise std used:              {gyro_std:.6e} rad/s")
    print()
    print(f"Final filtered omega estimate:   {ekf.x[3]:.6f} rad/s (true: {true_omega})")
    print(f"Final filtered theta:            {ekf.x[2]:.6f} rad (true: {true_theta:.6f})")
    print()
    print(f"RMSE filtered theta vs truth:     {filtered_theta_rmse:.6f} rad")
    print(f"RMSE raw-gyro-integration theta:  {raw_theta_rmse:.6f} rad")
    print(f"RMSE filtered omega vs truth:     {filtered_omega_rmse:.6f} rad/s")
    print()
    print(f"Final P diagonal (state uncertainty): {np.diag(ekf.P)}")