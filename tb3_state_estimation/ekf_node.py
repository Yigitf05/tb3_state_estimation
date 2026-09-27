"""
ROS 2 node wrapping the validated DifferentialDriveEKF.

Subscribes to /odom and /imu, runs the same predict()/update_gyro() calls
already validated standalone (synthetic data) and against a real recorded
bag, and publishes the fused pose as a nav_msgs/Odometry on
/odometry/filtered -- same message type as raw /odom, so it can be
compared directly (echoed, plotted, or viewed in RViz alongside /odom).

Design notes worth knowing before you read the code:

- predict() runs on every /odom message (using its twist.linear.x as the
  trusted control input v), update_gyro() runs on every /imu message.
  This matches the predict/update cadence from the design stage: /odom at
  ~50 Hz drives the motion model, /imu at ~200 Hz corrects omega in
  between.

- dt for predict() is computed from the INCOMING MESSAGE'S OWN header
  timestamp (msg.header.stamp), not this node's wall-clock time. Gazebo
  stamps /odom and /imu with simulated time already, so using the message
  timestamps directly means this node never needs a use_sim_time parameter
  or its own clock subscription to stay correct in sim -- it inherits
  whatever time the sensor data itself was stamped with. This is the same
  approach used in the bag-replay validation script, which is exactly why
  that validation is a meaningful predictor of this node's behavior.

- Publishing nav_msgs/Odometry (not a bare PoseStamped) means the
  covariance from the filter's P matrix rides along on the message, and
  RViz / any downstream consumer already knows how to interpret it the
  same way it interprets raw /odom.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from tb3_state_estimation.differential_drive_ekf import DifferentialDriveEKF

from tb3_state_estimation.differential_drive_ekf import (
    DifferentialDriveEKF,
    yaw_to_quaternion,
)

# Expected /odom rate measured earlier with `ros2 topic hz /odom` (~50 Hz).
# Used only to size the max-dt guard below -- not a control parameter.
EXPECTED_ODOM_PERIOD_SEC = 1.0 / 50.0
MAX_DT_MULTIPLIER = 5.0
MAX_DT_SEC = MAX_DT_MULTIPLIER * EXPECTED_ODOM_PERIOD_SEC  # 0.1 s



class EkfNode(Node):
    def __init__(self):
        super().__init__("ekf_node")

        self.ekf = DifferentialDriveEKF(initial_state=[0.0, 0.0, 0.0, 0.0])

        self.last_odom_time = None
        self.last_v = 0.0

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos_profile_sensor_data)
        self.imu_sub = self.create_subscription(Imu, "/imu", self.imu_callback, qos_profile_sensor_data)
        self.fused_pub = self.create_publisher(Odometry, "/odometry/filtered", 10)

        self.get_logger().info("EKF node started, waiting for /odom and /imu...")

    def odom_callback(self, msg: Odometry) -> None:
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        v = msg.twist.twist.linear.x
        self.last_v = v

        if self.last_odom_time is not None:
            dt = t - self.last_odom_time
            if dt <= 0.0:
                # Matches the same guard used in the bag-replay script --
                # should not happen with clean data, but never feed a bad
                # dt into predict() silently.
                self.get_logger().warn(f"Non-positive dt={dt:.4f}, skipping predict step")
                self.last_odom_time = t
                return

            if dt > MAX_DT_SEC:
                # Symmetric guard: an unusually large gap (startup lag, a
                # dropped connection, a stall) would blow up the linearized
                # motion model rather than represent a real large motion --
                # F's off-diagonal terms and the Q scaling both grow with
                # dt, so skip the predict step rather than trust it.
                self.get_logger().warn(
                    f"dt={dt:.4f}s exceeds MAX_DT_SEC={MAX_DT_SEC:.4f}s, "
                    "skipping predict step (stale/late odom message?)")
                self.last_odom_time = t

                return

            self.ekf.predict(v=v, dt=dt)
            self.publish_fused_odom(msg.header.stamp)

        self.last_odom_time = t

    def imu_callback(self, msg: Imu) -> None:
        self.ekf.update_gyro(msg.angular_velocity.z)

    def publish_fused_odom(self, stamp) -> None:
        x, y, theta = self.ekf.pose
        qx, qy, qz, qw = yaw_to_quaternion(theta)

        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = "odom"
        out.child_frame_id = "base_footprint"

        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        # 6x6 covariance, flattened -- only x, y, and yaw are meaningful
        # here since this is a planar filter. Index math: row-major 6x6,
        # so (x,x)=0, (y,y)=7, (yaw,yaw)=35.
        cov = [0.0] * 36
        cov[0] = float(self.ekf.P[0, 0])
        cov[7] = float(self.ekf.P[1, 1])
        cov[35] = float(self.ekf.P[2, 2])
        out.pose.covariance = cov

        out.twist.twist.linear.x = self.last_v
        out.twist.twist.angular.z = float(self.ekf.x[3])

        self.fused_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
