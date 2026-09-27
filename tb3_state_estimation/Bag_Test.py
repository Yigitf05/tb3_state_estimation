"""
Replay a recorded rosbag through the standalone DifferentialDriveEKF to
validate it against real Gazebo data, before any rclpy node exists.

This deliberately does NOT use rclpy nodes/subscribers -- it reads the bag
file directly and feeds messages into the same predict()/update_gyro()
class you already validated against synthetic data. The point is to catch
any integration bugs (timestamp handling, units, message field mistakes)
against real recorded data while it's still just a script you can rerun
and debug freely, before wiring it into a live ROS 2 node.

Usage:
    python3 test_against_bag.py <path_to_bag_folder>

Run this from a terminal where ROS 2 is already sourced (your .bashrc does
this automatically) since it needs rosbag2_py and the message types.
matplotlib is required for the plot: pip install matplotlib --user if it's
not already on the system.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from differential_drive_ekf import DifferentialDriveEKF


def read_bag(bag_path):
    # storage_id="" lets rosbag2_py auto-detect sqlite3 vs mcap from the
    # bag's own metadata.yaml -- if this fails on your install, check
    # `ros2 bag info <bag_path>` for the "Storage id" line and pass that
    # string explicitly instead.
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    messages = []
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        messages.append((t_ns, topic, msg))

    # Topics are interleaved in the bag but not guaranteed to already be
    # in strict global time order once read back -- sort explicitly.
    messages.sort(key=lambda entry: entry[0])
    return messages


def quat_to_yaw(q):
    """Standard quaternion -> yaw extraction for a planar robot."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test_against_bag.py <path_to_bag_folder>")
        sys.exit(1)

    bag_path = sys.argv[1]
    print(f"Reading bag: {bag_path}")
    messages = read_bag(bag_path)
    print(f"Loaded {len(messages)} messages")

    odom_count = sum(1 for _, topic, _ in messages if topic == "/odom")
    imu_count = sum(1 for _, topic, _ in messages if topic == "/imu")
    print(f"  /odom: {odom_count} messages")
    print(f"  /imu:  {imu_count} messages")
    if odom_count == 0 or imu_count == 0:
        print("ERROR: missing /odom or /imu in this bag -- check topic names with `ros2 bag info`")
        sys.exit(1)

    ekf = DifferentialDriveEKF(initial_state=[0.0, 0.0, 0.0, 0.0])

    last_odom_t = None
    dts = []

    filtered_xy = []
    raw_odom_xy = []
    filtered_theta = []
    raw_odom_theta = []

    for t_ns, topic, msg in messages:
        t_sec = t_ns / 1e9

        if topic == "/odom":
            v = msg.twist.twist.linear.x

            if last_odom_t is not None:
                dt = t_sec - last_odom_t
                if dt <= 0:
                    # Should never happen with clean data -- flag it
                    # rather than silently feeding a bad dt into predict().
                    print(f"WARNING: non-positive dt={dt:.6f} at t={t_sec:.3f}, skipping")
                    last_odom_t = t_sec
                    continue
                dts.append(dt)
                ekf.predict(v=v, dt=dt)

            last_odom_t = t_sec

            filtered_xy.append((ekf.x[0], ekf.x[1]))
            filtered_theta.append(ekf.x[2])
            raw_odom_xy.append((msg.pose.pose.position.x, msg.pose.pose.position.y))
            raw_odom_theta.append(quat_to_yaw(msg.pose.pose.orientation))

        elif topic == "/imu":
            ekf.update_gyro(msg.angular_velocity.z)

    if not dts:
        print("ERROR: never got two /odom messages in a row -- can't compute any dt")
        sys.exit(1)

    dts = np.array(dts)
    print()
    print("Predict-step dt statistics (seconds) -- sanity-check these against")
    print("the /odom publish rate you measured earlier with `ros2 topic hz /odom`:")
    print(f"  min={dts.min():.4f}  max={dts.max():.4f}  mean={dts.mean():.4f}")

    filtered_xy = np.array(filtered_xy)
    raw_odom_xy = np.array(raw_odom_xy)

    final_pos_diff = np.linalg.norm(filtered_xy[-1] - raw_odom_xy[-1])
    print()
    print(f"Final filtered position:  ({filtered_xy[-1][0]:.4f}, {filtered_xy[-1][1]:.4f})")
    print(f"Final raw odom position:  ({raw_odom_xy[-1][0]:.4f}, {raw_odom_xy[-1][1]:.4f})")
    print(f"Difference:                {final_pos_diff:.6f} m")
    print()
    print("Expected at this stage: filtered and raw should stay close together,")
    print("since Gazebo's odometry is still near-noiseless -- there's no wheel-slip")
    print("noise injected yet for the filter to meaningfully correct. What you're")
    print("checking for here is the absence of NaNs, blown-up values, or a filtered")
    print("trajectory that diverges from raw odom without explanation -- not a big")
    print("accuracy improvement yet.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(raw_odom_xy[:, 0], raw_odom_xy[:, 1], label="raw /odom", linewidth=2)
    axes[0].plot(filtered_xy[:, 0], filtered_xy[:, 1], label="EKF filtered", linestyle="--")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("Trajectory: raw odom vs filtered")
    axes[0].legend()
    axes[0].axis("equal")

    axes[1].plot(raw_odom_theta, label="raw /odom theta")
    axes[1].plot(filtered_theta, label="EKF filtered theta", linestyle="--")
    axes[1].set_xlabel("odom message index")
    axes[1].set_ylabel("theta (rad)")
    axes[1].set_title("Heading: raw odom vs filtered")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("ekf_bag_validation.png", dpi=150)
    print()
    print("Saved plot to ekf_bag_validation.png")


if __name__ == "__main__":
    main()