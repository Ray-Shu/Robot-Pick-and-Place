import sys
import rospy
import cv2
import cv_bridge
import numpy as np
import tf2_ros
import tf_conversions

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import TransformStamped, Transform


class CameraNode:
    """Basic Node that runs a camera on a given camera index and publishes to ros."""

    def __init__(self):
        """Initialize a camera node by getting the rosparam and opening the opencv stream."""

        rospy.init_node("camera_node", anonymous=True)
        cam_param = rospy.search_param("cam_idx")
        self.cam_idx = rospy.get_param(cam_param, None)

        if self.cam_idx is None:
            rospy.logwarn("Must pass camera index as _cam_idx:=<cam_idx>")
            exit()

        rospy.delete_param(cam_param)  # delete param so its needed for future runs.
        rospy.loginfo(f"Initialized Camera on topic /cameras/cam{self.cam_idx}")

        # setup rotation matrices
        self.x = rospy.get_param("~x", 0.0)
        self.y = rospy.get_param("~y", 0.0)
        self.z = rospy.get_param("~z", 0.0)
        self.roll = np.deg2rad(rospy.get_param("~roll", 0.0))
        self.pitch = np.deg2rad(rospy.get_param("~pitch", 0.0))
        self.yaw = np.deg2rad(rospy.get_param("~yaw", 0.0))
        roll_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.roll), -np.sin(self.roll)],
                [0, np.sin(self.roll), np.cos(self.roll)],
            ]
        )
        pitch_mat = np.array(
            [
                [np.cos(self.pitch), 0, np.sin(self.pitch)],
                [0, 1, 0],
                [-np.sin(self.pitch), 0, np.cos(self.pitch)],
            ]
        )
        yaw_mat = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw), 0],
                [np.sin(self.yaw), np.cos(self.yaw), 0],
                [0, 0, 1],
            ]
        )
        rotation_mat = yaw_mat @ pitch_mat @ roll_mat
        # rotation matrix quaternion
        rotation_mat = np.column_stack(
            [np.row_stack([rotation_mat, [0, 0, 0]]), [0, 0, 0, 1]]
        )
        self.quaternion = tf_conversions.transformations.quaternion_from_matrix(
            rotation_mat
        )

        # --- TF2 ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # --- Publishers ---
        self.image_pub = rospy.Publisher(
            f"/cameras/cam{self.cam_idx}", Image, queue_size=10
        )
        self.compressed_image_pub = rospy.Publisher(
            f"/cameras/cam{self.cam_idx}/compressed", CompressedImage, queue_size=10
        )

        self.tracked_image_pub = rospy.Publisher(
            f"/cameras/cam{self.cam_idx}/tracked_points", Image, queue_size=10
        )
        self.compressed_tracked_image_pub = rospy.Publisher(
            f"/cameras/cam{self.cam_idx}/tracked_points/compressed",
            CompressedImage,
            queue_size=10,
        )

        self.idx_pub = rospy.Publisher("/cam_idx", Int32, queue_size=1)
        self.compression_pub = rospy.Publisher("/compression", Bool, queue_size=1)

        # setup realsense if necessary
        self.is_rs = rospy.get_param("~is_rs")
        self.bridge = cv_bridge.CvBridge()
        if self.is_rs:
            self.rs_depth_sub = rospy.Subscriber(
                f"/camera{self.cam_idx}/aligned_depth_to_color/image_raw",
                Image,
                self.rs_depth_cb,
            )
            self.rs_image_sub = rospy.Subscriber(
                f"/camera{self.cam_idx}/color/image_raw", Image, self.rs_img_cb
            )
            self.depth_pub = rospy.Publisher(
                f"/cameras/cam{self.cam_idx}/aligned_depth", Image, queue_size=10
            )
            self.compressed_depth_pub = rospy.Publisher(
                f"/cameras/cam{self.cam_idx}/aligned_depth/compressed",
                CompressedImage,
                queue_size=10,
            )
            rospy.loginfo(f"/camera{self.cam_idx}/color/image_raw")
        else:
            # setup opencv input
            self.video_cap = cv2.VideoCapture(self.cam_idx)
            # timer for image update
            rospy.Timer(rospy.Duration(0.05), self.update_callback)

    def rs_depth_cb(self, data):
        """RealSense data callback

        Args:
            data (Image): depth image from realsense
        """
        self.depth_pub.publish(data)

    def rs_img_cb(self, data):
        """Realsense image callback

        Args:
            data (Image): new image message
        """
        c_data = self.bridge.cv2_to_compressed_imgmsg(
            self.bridge.imgmsg_to_cv2(data, "rgb8")
        )
        self.update(data, c_data)

    def update_callback(self, event=None):
        """Callback to update the image on the camera topic."""
        ret, frame = self.video_cap.read()
        if not ret:
            rospy.logwarn(f"Couldn't get image frame for camera {self.cam_idx}")
            return

        # convert image color format and publish
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        c_image: CompressedImage = self.bridge.cv2_to_compressed_imgmsg(cvim=frame)
        image: Image = self.bridge.cv2_to_imgmsg(cvim=frame, encoding="rgb8")
        image.header.stamp = rospy.get_rostime()
        self.update(image, c_image)

    def update(self, image: Image, c_image: CompressedImage):
        """Update published data with new image message

        Args:
            image (Image): image message
        """
        # publish index
        self.idx_pub.publish(Int32(data=self.cam_idx))

        image.header.frame_id = f"{self.cam_idx}"
        self.image_pub.publish(image)
        self.compressed_image_pub.publish(c_image)

        # publish transform to camera pose
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base_link"
        t.child_frame_id = f"cam{self.cam_idx}"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.z
        t.transform.rotation.x = self.quaternion[0]
        t.transform.rotation.y = self.quaternion[1]
        t.transform.rotation.z = self.quaternion[2]
        t.transform.rotation.w = self.quaternion[3]
        self.tf_broadcaster.sendTransform(t)


def main(args):
    rospy.loginfo("Starting a CameraNode ...")
    node = CameraNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Camera...")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
