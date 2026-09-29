"""
Compare the LIVE ekf_node's published /odometry/filtered against raw
/odom, both captured in the same bag while actually driving.

This is a different (and more conclusive) test than the earlier
test_against_bag.py: that script re-ran the standalone EKF class offline
against recorded topics. This script does NOT touch the EKF class at all
-- it just reads two already-published live streams and compares them
directly, which is what actually proves the DEPLOYED NODE behaves
correctly end to end, not just the math in isolation.

Alignment trick: ekf_node.py's publish_fused_odom() deliberately reuses
the exact same header.stamp from the /odom message that triggered that
predict() call (see ekf_node.py). So every /odometry/filtered message's
timestamp exactly matches one specific /odom message's timestamp -- this
script aligns the two streams by that exact timestamp match, no
interpolation or nearest-neighbor guessing required.

Usage:
    python3 compare_live_ekf.py <path_to_bag_folder>

Run from a sourced ROS 2 terminal (needs rosbag2_py + message types).
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_bag(bag_path):
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

    messages.sort(key=lambda entry: entry[0])
    return messages


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def header_stamp_ns(msg) -> int:
    """Nanoseconds from the MESSAGE's OWN header.stamp field -- not the bag
    recorder's arrival timestamp (reader.read_next()'s t), which is a
    different clock per topic and does not line up across topics. ekf_node
    deliberately copies /odom's header.stamp through unchanged onto
    /odometry/filtered, so this is what actually lets the two streams be
    matched exactly."""
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 compare_live_ekf.py <path_to_bag_folder>")
        sys.exit(1)

    bag_path = sys.argv[1]
    print(f"Reading bag: {bag_path}")
    messages = read_bag(bag_path)
    print(f"Loaded {len(messages)} messages")

    odom_count = sum(1 for _, topic, _ in messages if topic == "/odom")
    filtered_count = sum(1 for _, topic, _ in messages if topic == "/odometry/filtered")
    print(f"  /odom:               {odom_count} messages")
    print(f"  /odometry/filtered:  {filtered_count} messages")

    if odom_count == 0 or filtered_count == 0:
        print("ERROR: need both /odom and /odometry/filtered in this bag")
        print("(check topic names with `ros2 bag info <bag_path>`, and make sure")
        print(" ekf_node was actually running and recording during capture)")
        sys.exit(1)

    # Index filtered messages by their MESSAGE-INTERNAL timestamp (see
    # header_stamp_ns docstring -- this is not the same as the bag
    # reader's per-message arrival time).
    filtered_by_stamp = {}
    for _, topic, msg in messages:
        if topic == "/odometry/filtered":
            filtered_by_stamp[header_stamp_ns(msg)] = msg

    raw_xy, raw_theta = [], []
    filt_xy, filt_theta = [], []
    filt_cov = []  # (var_x, var_y, var_yaw) over time
    times = []

    unmatched = 0

    for _, topic, msg in messages:
        if topic != "/odom":
            continue

        key = header_stamp_ns(msg)
        matched = filtered_by_stamp.get(key)
        if matched is None:
            # Expected for the very first /odom message (predict() only
            # starts producing output from the second /odom message
            # onward, since the first just establishes the dt baseline).
            unmatched += 1
            continue

        raw_xy.append((msg.pose.pose.position.x, msg.pose.pose.position.y))
        raw_theta.append(quat_to_yaw(msg.pose.pose.orientation))

        filt_xy.append((matched.pose.pose.position.x, matched.pose.pose.position.y))
        filt_theta.append(quat_to_yaw(matched.pose.pose.orientation))
        filt_cov.append((matched.pose.covariance[0], matched.pose.covariance[7], matched.pose.covariance[35]))
        times.append(key / 1e9)

    print(f"\nAligned pairs: {len(raw_xy)}  (unmatched /odom messages: {unmatched})")
    if len(raw_xy) == 0:
        print("ERROR: no aligned pairs found -- something is wrong with the timestamps")
        sys.exit(1)

    raw_xy = np.array(raw_xy)
    filt_xy = np.array(filt_xy)
    raw_theta = np.array(raw_theta)
    filt_theta = np.array(filt_theta)
    filt_cov = np.array(filt_cov)
    times = np.array(times)
    times -= times[0]  # start plot at t=0 seconds

    pos_err = np.linalg.norm(raw_xy - filt_xy, axis=1)
    theta_err = np.abs(np.array([wrap_angle(a - b) for a, b in zip(raw_theta, filt_theta)]))

    print(f"Position difference (raw vs filtered): mean={pos_err.mean():.4f} m, max={pos_err.max():.4f} m")
    print(f"Heading difference  (raw vs filtered): mean={theta_err.mean():.6f} rad, max={theta_err.max():.6f} rad")
    print()
    print("Expected at this stage: small differences, same reasoning as before --")
    print("Gazebo's odometry is still near-noiseless, so there isn't much for the")
    print("live node to correct yet either. What matters here is that the DEPLOYED")
    print("node's output matches what we already validated offline -- no surprises")
    print("from the ROS wiring itself (QoS, timestamps, message handling).")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(raw_xy[:, 0], raw_xy[:, 1], label="raw /odom", linewidth=2)
    axes[0].plot(filt_xy[:, 0], filt_xy[:, 1], label="live /odometry/filtered", linestyle="--")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("Trajectory: raw odom vs live filtered")
    axes[0].legend()
    axes[0].axis("equal")

    axes[1].plot(times, raw_theta, label="raw /odom theta")
    axes[1].plot(times, filt_theta, label="live filtered theta", linestyle="--")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("theta (rad)")
    axes[1].set_title("Heading: raw odom vs live filtered")
    axes[1].legend()

    axes[2].plot(times, filt_cov[:, 0], label="var(x)")
    axes[2].plot(times, filt_cov[:, 1], label="var(y)")
    axes[2].plot(times, filt_cov[:, 2], label="var(yaw)")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("variance")
    axes[2].set_title("Filter uncertainty (P diagonal) over time")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("live_ekf_validation.png", dpi=150)
    print("\nSaved plot to live_ekf_validation.png")


if __name__ == "__main__":
    main()
