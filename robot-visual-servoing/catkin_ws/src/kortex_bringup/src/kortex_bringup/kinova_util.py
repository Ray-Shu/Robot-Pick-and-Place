import rospy
import numpy as np
import tf2_ros
import tf_conversions

from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped


class KinovaUtil:
    """Class that provides utility functions for the Kinova arm related"""

    def __init__(self):
        """Initialize the KinovaUtil by setting up all required buffers."""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_eef_pose(self, eef_frame="tool_frame"):
        """Gets the end effector pose of the kinova arm as a vector.

        Returns:
            ArrayLike: x, y, z, roll, pitch, yaw of the tool frame (end effector)
        """
        try:
            eef_transform: TransformStamped = self.tf_buffer.lookup_transform(
                "base_link", eef_frame, rospy.Time(0)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(
                f"Failed to lookup transform from tool_frame to base_link: {e}"
            )
            return None

        tf = eef_transform.transform
        tr = tf.translation
        rot = tf.rotation
        roll, pitch, yaw = tf_conversions.transformations.euler_from_quaternion(
            [rot.x, rot.y, rot.z, rot.w]
        )

        return np.array([tr.x, tr.y, tr.z, roll, pitch, yaw])

    def get_arm_joints(self):
        """Gets the 7 arm joints of the robot, excluding joints related to the robot fingers.

        Note this call will block until the joints can be obtained.

        Returns:
            ArrayLike: 7 joint angles in radians.
        """
        data = rospy.wait_for_message("/my_gen3/joint_states", JointState)
        current_joints = list(data.position[:7])
        return np.array(current_joints)
