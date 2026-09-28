"""
ROS 2 node wrapping the validated DifferentialDriveEKF.

Subscribes to odometry and IMU, runs the same predict()/update_gyro() calls
validated offline (synthetic data, bag replay, noise injection), and
publishes the fused pose as nav_msgs/Odometry on odometry/filtered.

Topic names are relative and chosen by remapping, as in odom_noise_relay.
The node does not know whether its odometry is clean or corrupted:

    ros2 run tb3_state_estimation ekf_node --ros-args -r odom:=odom_noisy

Without the remap it runs on clean /odom. The results look almost the same
either way (heading comes from the gyro), so check the resolved topic in
the startup log or with `ros2 node info /ekf_node`, never from the plots.

Design notes:

- Initialization matches run_filter() in test_noisy_odom.py. The first
  odometry message sets the state [x, y, yaw, omega] from its pose and
  twist, with P = diag(1e-8, 1e-8, 1e-8, 1e-3). No predict and no publish
  on that message. IMU messages are dropped until then.

- predict() runs on every later odometry message, with its twist.linear.x
  as the control input v. update_gyro() runs on every IMU message. Messages
  are processed in arrival order; offline replay sorts by stamp. At equal
  stamps that is a time shift of at most one IMU period (5 ms), accepted.

- dt comes from each message's own header.stamp, converted exactly as
  load_bag() does, so live and offline dt values are identical doubles.
  No use_sim_time is needed.

- Subscriptions are best effort (sensor data profile), compatible with
  reliable and best-effort publishers alike. The publisher is reliable
  depth 10 for RViz and rosbag2.

- No TF is published. odom -> base_footprint has exactly one owner, the
  simulator's odometry.
"""

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from tb3_state_estimation.differential_drive_ekf import (
    DifferentialDriveEKF,
    quaternion_to_yaw,
    yaw_to_quaternion,
)

# Expected odometry rate measured with `ros2 topic hz /odom` (~50 Hz).
# Used only to size the max-dt guard, same value as test_noisy_odom.py.
EXPECTED_ODOM_PERIOD_SEC = 1.0 / 50.0
MAX_DT_SEC = 5.0 * EXPECTED_ODOM_PERIOD_SEC  # 0.1 s

# Same as run_filter(): the odom frame is defined by the start pose, so
# x, y, yaw are known essentially exactly; omega is corrected by the first
# gyro update anyway.
INITIAL_COVARIANCE_DIAG = (1e-8, 1e-8, 1e-8, 1e-3)


def stamp_to_sec(stamp) -> float:
    """Same conversion as load_bag() in test_noisy_odom.py."""
    return stamp.sec + stamp.nanosec * 1e-9


class EkfNode(Node):
    def __init__(self):
        super().__init__("ekf_node")

        # Created from the first odometry message, see odom_callback().
        self.ekf = None
        self.last_odom_time = None
        self.last_v = 0.0

        self.odom_sub = self.create_subscription(
            Odometry, "odom", self.odom_callback, qos_profile_sensor_data)
        self.imu_sub = self.create_subscription(
            Imu, "imu", self.imu_callback, qos_profile_sensor_data)
        self.fused_pub = self.create_publisher(Odometry, "odometry/filtered", 10)

        # Resolved names, after remapping. Read this line every run.
        self.get_logger().info(
            f"Subscribed to {self.odom_sub.topic_name} and {self.imu_sub.topic_name}, "
            f"publishing {self.fused_pub.topic_name}, "
            f"initial P diag {INITIAL_COVARIANCE_DIAG}, max_dt {MAX_DT_SEC:.3f} s")
        self.get_logger().info("Waiting for the first odometry message...")

    def odom_callback(self, msg: Odometry) -> None:
        t = stamp_to_sec(msg.header.stamp)
        v = msg.twist.twist.linear.x
        self.last_v = v

        if self.ekf is None:
            self.initialize(msg, t)
            return

        # Same order as run_filter(): the stamp always advances, then the
        # guards decide whether this interval is integrated.
        dt = t - self.last_odom_time
        self.last_odom_time = t

        if dt <= 0.0:
            self.get_logger().warn(
                f"Non-positive dt={dt:.4f} s, skipping predict step",
                throttle_duration_sec=1.0)
            return

        if dt > MAX_DT_SEC:
            # A large gap (startup lag, stall, dropped connection) would be
            # integrated as one long linearized step. Skip it instead.
            self.get_logger().warn(
                f"dt={dt:.4f} s exceeds {MAX_DT_SEC:.3f} s, skipping predict step",
                throttle_duration_sec=1.0)
            return

        self.ekf.predict(v=v, dt=dt)
        self.publish_fused_odom(msg.header.stamp)

    def initialize(self, msg: Odometry, t: float) -> None:
        """State from the first odometry message, like run_filter()."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        omega = msg.twist.twist.angular.z

        self.ekf = DifferentialDriveEKF(
            initial_state=[p.x, p.y, yaw, omega],
            initial_covariance=np.diag(INITIAL_COVARIANCE_DIAG),
        )
        self.last_odom_time = t
        self.get_logger().info(
            f"Initialized from first odometry message at t={t:.3f} s: "
            f"x={p.x:.4f} m, y={p.y:.4f} m, yaw={np.degrees(yaw):.3f} deg, "
            f"omega={omega:.4f} rad/s")

    def imu_callback(self, msg: Imu) -> None:
        # Nothing to correct before the state exists. run_filter() likewise
        # drops IMU messages stamped before the first odometry message.
        if self.ekf is None:
            return
        self.ekf.update_gyro(msg.angular_velocity.z)

    def publish_fused_odom(self, stamp) -> None:
        x, y, theta = self.ekf.pose
        qx, qy, qz, qw = yaw_to_quaternion(theta)

        out = Odometry()
        out.header.stamp = stamp            # same stamp as the input message
        out.header.frame_id = "odom"
        out.child_frame_id = "base_footprint"

        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        # 6x6 covariance, row-major: (x,x)=0, (y,y)=7, (yaw,yaw)=35.
        # Only these are meaningful for a planar filter.
        cov = [0.0] * 36
        cov[0] = float(self.ekf.P[0, 0])
        cov[7] = float(self.ekf.P[1, 1])
        cov[35] = float(self.ekf.P[2, 2])
        out.pose.covariance = cov

        out.twist.twist.linear.x = float(self.last_v)
        out.twist.twist.angular.z = float(self.ekf.x[3])

        self.fused_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
