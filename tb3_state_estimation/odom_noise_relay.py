"""
Relay node: clean /odom in, wheel-level corrupted /odom_noisy out.

All noise and integration math lives in WheelOdomCorruptor (wheel_noise.py),
which is verified bit-identical to corrupt_odometry() in test_noisy_odom.py
(check_wheel_noise.py). This node only does ROS plumbing:

- Parameters are declared read-only and read once at startup. Changing the
  noise model mid-run would silently turn one experiment into two.
- The RNG lives here. Noise is drawn once per incoming /odom message, per
  wheel, because the offline model and the filter's v_noise_ratio both
  assume per-message noise at the odom rate.
- header.stamp, frame_id and child_frame_id are copied from the input. The
  noisy message describes the same instant, in the same frames, as the
  clean one. No clock is used, so no use_sim_time is needed.
- No TF is published. odom -> base_footprint has exactly one owner (the
  simulator's odometry). A second broadcaster would make TF alternate
  between clean and noisy poses and break Nav2 and RViz.

QoS: subscribe best effort (sensor data profile, compatible with any
publisher), publish reliable depth 10 (compatible with best-effort and
reliable subscribers alike, including RViz and rosbag2).

Run:
    ros2 run tb3_state_estimation odom_noise_relay --ros-args --params-file relay_default.yaml
"""

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from nav_msgs.msg import Odometry

from tb3_state_estimation.differential_drive_ekf import quaternion_to_yaw, yaw_to_quaternion
from tb3_state_estimation.wheel_noise import DEFAULT_MAX_DT_SEC, WheelOdomCorruptor


def stamp_to_sec(stamp) -> float:
    """Same conversion as load_bag() in test_noisy_odom.py, so live and
    offline dt values are identical doubles."""
    return stamp.sec + stamp.nanosec * 1e-9


# (name, default, description). The default's Python type fixes the
# parameter type: 0.0 declares a double, 0 an integer.
PARAMETERS = [
    ("mismatch", 0.005, "Wheel diameter mismatch m; s_r = +m/2, s_l = -m/2"),
    ("sigma", 0.03, "White multiplicative noise std per wheel per message"),
    ("common_scale", 0.0, "Common scale error c on both wheels (0.01 = 1%)"),
    ("wheel_separation", 0.160, "Wheel separation b [m]"),
    ("seed", 0, "RNG seed (non-negative integer)"),
    ("max_dt", DEFAULT_MAX_DT_SEC, "Skip integration when dt exceeds this [s]"),
]


class OdomNoiseRelay(Node):
    def __init__(self):
        super().__init__("odom_noise_relay")

        for name, default, text in PARAMETERS:
            self.declare_parameter(
                name, default, ParameterDescriptor(description=text, read_only=True))
        p = {name: self.get_parameter(name).value for name, _, _ in PARAMETERS}

        if p["sigma"] < 0.0:
            raise ValueError("sigma must be non-negative")
        if p["seed"] < 0:
            raise ValueError("seed must be a non-negative integer")

        self.sigma = float(p["sigma"])
        self.rng = np.random.default_rng(p["seed"])
        # The class validates wheel_separation and max_dt itself.
        self.corr = WheelOdomCorruptor(
            wheel_separation=p["wheel_separation"],
            mismatch=p["mismatch"],
            common_scale=p["common_scale"],
            max_dt=p["max_dt"],
        )

        self.sub = self.create_subscription(
            Odometry, "odom", self.odom_callback, qos_profile_sensor_data)
        self.pub = self.create_publisher(Odometry, "odom_noisy", 10)

        self.get_logger().info(
            "Relay config: " + ", ".join(f"{k}={v}" for k, v in p.items()))
        self.get_logger().info(
            f"Subscribed to {self.sub.topic_name}, publishing {self.pub.topic_name}")

    def odom_callback(self, msg: Odometry) -> None:
        t = stamp_to_sec(msg.header.stamp)
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        n_r, n_l = self.rng.normal(0.0, self.sigma, 2)

        if not self.corr.initialized:
            # Start from the clean pose, like track[0] offline.
            q = msg.pose.pose.orientation
            self.corr.initialize(
                t, msg.pose.pose.position.x, msg.pose.pose.position.y,
                quaternion_to_yaw(q.x, q.y, q.z, q.w))
            v_n, w_n = self.corr.corrupt_twist(v, w, n_r, n_l)
            self.get_logger().info(f"Initialized from first odom message at t={t:.3f} s")
        else:
            v_n, w_n, dt = self.corr.step(t, v, w, n_r, n_l)
            if not (0.0 < dt <= self.corr.max_dt):
                self.get_logger().warn(
                    f"dt={dt:.4f} s outside (0, {self.corr.max_dt:.3f}], "
                    "pose held for this message",
                    throttle_duration_sec=1.0)

        self.publish(msg, v_n, w_n)

    def publish(self, msg: Odometry, v_n: float, w_n: float) -> None:
        x, y, theta = self.corr.pose
        qx, qy, qz, qw = yaw_to_quaternion(theta)

        out = Odometry()
        out.header = msg.header                  # stamp and frame_id, not now()
        out.child_frame_id = msg.child_frame_id

        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.position.z = msg.pose.pose.position.z
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        # Covariances copied unchanged; nothing downstream reads them.
        out.pose.covariance = msg.pose.covariance

        out.twist.twist.linear.x = float(v_n)
        out.twist.twist.angular.z = float(w_n)
        out.twist.covariance = msg.twist.covariance

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomNoiseRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
