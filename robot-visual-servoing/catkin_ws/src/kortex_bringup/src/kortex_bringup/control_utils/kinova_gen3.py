#!/usr/bin/python3
###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed 
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

from kortex_driver.srv import *
from kortex_driver.msg import *


# #####
# Limits for Gen3 Joints
# In degree
# #####

JOINT_LIMIT_DEGREES = {
    0: [-180.0, 180.0],
    1: [-128.9, 128.9],
    2: [-180.0, 180.0],
    3: [-147.8, 147.8],
    4: [-180.0, 180.0],
    5: [-120.3, 120.3],
    6: [-180.0, 180.0],
}

# #####
# Limits for Gen3 Joints
# In radians
# #####

JOINT_LIMIT = {
    0: [-3.141592653589793, 3.141592653589793],
    1: [-2.2497294058206907, 2.2497294058206907],
    2: [-3.141592653589793, 3.141592653589793],
    3: [-2.5795966344476193, 2.5795966344476193],
    4: [-3.141592653589793, 3.141592653589793],
    5: [-2.0996310901491784, 2.0996310901491784],
    6: [-3.141592653589793, 3.141592653589793],
    7: [0, 1]
}

# #####
# DH parameters
# #####

DH_PARAMETERS = {
    0: [  np.pi, 0,                  0,     0],
    1: [np.pi/2, 0, -(0.1564 + 0.1284),     0],
    2: [np.pi/2, 0, -(0.0054 + 0.0064), np.pi],
    3: [np.pi/2, 0, -(0.2104 + 0.2104), np.pi],
    4: [np.pi/2, 0, -(0.0064 + 0.0064), np.pi],
    5: [np.pi/2, 0, -(0.2084 + 0.1059), np.pi],
    6: [np.pi/2, 0,                  0, np.pi],
	7: [  np.pi, 0, -(0.1059 + 0.0615), np.pi],
	8: [      0, 0,             (0.140),     0]
}

JOINT_NAME_TO_ID = {
    "joint_1": 0,
    "joint_2": 1,
    "joint_3": 2,
    "joint_4": 3,
    "joint_5": 4,
    "joint_6": 5,
    "joint_7": 6,
}

# #####
# Kinova Gen3 Object
# #####


class KinovaGen3(object):
    """Kinova Gen3.
    Interact with robot control.
    """
    def __init__(
        self,
        robot_name: str = "my_gen3",
        ):
        # ####################
        # Connect to Gen3 and setup publishers and subscribers
        # ####################
        try:
            # Gen3 Parameters
            # -----
            self.robot_name = rospy.get_param('~robot_name', robot_name)
            self.dof = rospy.get_param("/{}/kortex_driver/dof".format(self.robot_name))
            self.joint_names = rospy.get_param("/{}/kortex_driver/joint_names".format(self.robot_name))
            self.is_gripper_present = rospy.get_param("/{}/is_gripper_present".format(self.robot_name), True)
            self.HOME_ACTION_IDENTIFIER = 2 # The Home Action is used to home the robot. It cannot be deleted and is always ID #2
            self.JOINT_NAME_TO_ID = JOINT_NAME_TO_ID
            self.JOINT_LIMIT = JOINT_LIMIT

            self.prev_gripper_cmd = None
            self.prev_cv_cmd = None

            self.movement_blocked = False

            # Gen3 services
            # -----
            # Action topic subscriber
            self.last_action_notif_type = None
            self.action_topic_sub = rospy.Subscriber("/{}/action_topic".format(self.robot_name), 
                                        ActionNotification, self._action_topic_cb)

            self._init_gen3_services()


            # Gen3 subscribers
            # -----
            # Store robot pose into python
            self.position = None
            self.vel = None
            self.cartesian_pose = None
            self.joint_state_sub = rospy.Subscriber("/{}/base_feedback/joint_state".format(self.robot_name), 
                                            JointState, self._joint_state_cb)
            
            
            self.base_feedback_sub = rospy.Subscriber("/{}/base_feedback".format(self.robot_name), 
                                            BaseCyclic_Feedback, self._base_feedback_cb)

            
            # Gen3 publishers
            # -----
            #self.cartesian_vel_pub = rospy.Publisher("/{}/in/cartesian_velocity".format(self.robot_name), 
            #                            TwistCommand, queue_size=10)
            
            #self.joint_vel_pub = rospy.Publisher("/{}/in/joint_velocity".format(self.robot_name), 
            #                            Base_JointSpeeds, queue_size=10)
            self.joint_velocity_pub = rospy.Publisher("/" + self.robot_name + "/in/joint_velocity", Base_JointSpeeds, queue_size=1)
            self.cartesian_velocity_pub = rospy.Publisher("/" + self.robot_name + "/in/cartesian_velocity", TwistCommand, queue_size=1)
            
            # Further initialization
            self._call_clear_faults()
            self._call_subscribe_to_a_robot_notification()

            self.going_home = False # bool used to prevent joint velocity commands during go_home execution
            self.home_button_pressed = False
            self.just_went_home = False

        except rospy.ServiceException:
            rospy.logerr("Failed to initialize Kinova Gen3, {}!".format(self.robot_name))
            rospy.signal_shutdown("Exiting...")

    def open_gripper(self):
        while self.position[7] > 0.07:
            self.send_gripper_command(0.3)
        self.send_gripper_command(0.0)

    def _init_gen3_services(self):
        """Initialize Gen3 services.
        """
        try:
            # Clear Faults
            clear_faults_full_name = '/{}/base/clear_faults'.format(self.robot_name)
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            # For Joint Angles
            # -----
            # Read action
            read_action_full_name = '/{}/base/read_action'.format(self.robot_name)
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

            # Execute action
            execute_action_full_name = '/{}/base/execute_action'.format(self.robot_name)
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

            # Set Cartesian Reference frame
            set_cartesian_reference_frame_full_name = '/{}/control_config/set_cartesian_reference_frame'.format(self.robot_name)
            rospy.wait_for_service(set_cartesian_reference_frame_full_name)
            self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name, SetCartesianReferenceFrame)

            # Gripper command
            send_gripper_command_full_name = '/{}/base/send_gripper_command'.format(self.robot_name)
            rospy.wait_for_service(send_gripper_command_full_name)
            self.send_gripper_command_srv = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

            # Activate publishing of action notification
            activate_publishing_of_action_notification_full_name = '/{}/base/activate_publishing_of_action_topic'.format(self.robot_name)
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)
        
            get_product_configuration_full_name = '/{}/base/get_product_configuration'.format(self.robot_name)
            rospy.wait_for_service(get_product_configuration_full_name)
            self.get_product_configuration = rospy.ServiceProxy(get_product_configuration_full_name, GetProductConfiguration)

            validate_waypoint_list_full_name = '/{}/base/validate_waypoint_list'.format(self.robot_name)
            rospy.wait_for_service(validate_waypoint_list_full_name)
            self.validate_waypoint_list = rospy.ServiceProxy(validate_waypoint_list_full_name, ValidateWaypointList)

            get_cartesian_full_name = '/{}/base/get_measured_cartesian_pose'.format(self.robot_name)
            rospy.wait_for_service(get_cartesian_full_name)
            self.get_cartesian_pose = rospy.ServiceProxy(get_cartesian_full_name, GetMeasuredCartesianPose)


        except rospy.ServiceException:
            rospy.logerr("Failed to initialize Kinova Gen3 services!")

    def _action_topic_cb(self, notif):
        """Monitor kinova action notification.
        """
        self.last_action_notif_type = notif.action_event
        self.last_action_notif = notif

    def convert_pose_to_array(self, pose_message):
        keys = ['tool_pose_x', 'tool_pose_y', 'tool_pose_z', 'tool_pose_theta_x', 'tool_pose_theta_y', 'tool_pose_theta_z']
        pose_array = np.zeros(6)
        for i, key in enumerate(keys):
            pose_array[i] = getattr(pose_message, key)
        
        return pose_array

    def _base_feedback_cb(self, msg):
        self.cartesian_pose = self.convert_pose_to_array(msg.base)

    def _joint_state_cb(self, msg):
        """Store joint angles inside the class instance.
        """

        self.position = np.array(msg.position).astype(np.float64)
        self.vel = np.array(msg.velocity[:len(self.joint_names)]).astype(np.float64)
 

    def _call_subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        rospy.sleep(1.0)
        return True

    def _call_clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            rospy.sleep(2.5)
            return True

    # #####
    # Full Arm Movement
    # #####

    def block_movement(self):
        self.movement_blocked = True

    def unblock_movement(self):
        self.movement_blocked = False

    def _wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                abort_reason = getattr(self.last_action_notif, 'status', 'Unknown reason')
                rospy.loginfo("Received ACTION_ABORT notification")
                rospy.loginfo(f"ACTION_ABORT Reason: {abort_reason}")
                return False
            else:
                time.sleep(0.01)

    def go_home(self):
        """Send Gen3 to default home position.
        """
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        
        if self.going_home:
            return False
        self.send_joint_speeds_command(np.zeros(7))
        self.going_home = True
        rospy.sleep(0.5)
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        self.last_action_notif_type = None
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        # Execute the HOME action if we could read it
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot home...")
            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ExecuteAction")
                return False
            else:
                success =  self._wait_for_action_end_or_abort()
                self.going_home = False
                self.home_button_pressed = False
                self.just_went_home = True
                print('Done going home.')
                return success
                

    def send_joint_angles(
        self,
        angles: list,
        angular_duration: float = 0.0,
        MAX_ANGULAR_DURATION: float = 30.0,
        ):
        """Move Gen3 to specified joint angles.
        Args:
            angles: list, 7 DOF, in degrees.
            angular_duration: float. Control duration between AngularWaypoint 
                in a trajectory. 0 by default.
            MAX_ANGULAR_DURATION: float. To validate if angles are safe.
        """
        # NOTE: IMPORTANT!
        # Beforehand, make sure joint velocity is 0.0
        # TODO: THIS IS CAUSING SOME WEIRD BUGS
        # [ERROR] [1650665544.432314487]: Attempt to get goal status on an uninitialized ServerGoalHandle or one that has no ActionServer associated with it.
        # self.send_joint_velocities([0, 0, 0, 0, 0, 0, 0])

        # Make sure angles is a numpy array
        if isinstance(angles, list):
            angles = np.array(angles)

        # Clip the degrees by joint_limit
        # for i in range(len(angles)):
        #     angles[i] = np.clip(angles[i], a_min=JOINT_LIMIT[i][0], a_max=JOINT_LIMIT[i][1])

        # Initialization
        self.last_action_notif_type = None
        req = ExecuteActionRequest()

        # Angles to send the arm
        # Use AngularWaypoint, per Kinova official example
        angular_waypoint = AngularWaypoint()
        # print(self.robot_name)
        # print(self.dof)
        try:
            # print(len(angles))
            
            assert len(angles) == self.dof
            
        except:
            rospy.logerr("Invalid angles.")
            return False
        for i in range(len(angles)):
            angular_waypoint.angles.append(angles[i])
        #print(angular_waypoint)
        
        # Each AngularWaypoint needs a duration and the global duration (from WaypointList) is disregarded. 
        # If you put something too small (for either global duration or AngularWaypoint duration), the trajectory will be rejected.
        angular_waypoint.duration = angular_duration

        # Initialize Waypoint and WaypointList
        trajectory = WaypointList()
        waypoint = Waypoint()
        #print(waypoint)
        waypoint.oneof_type_of_waypoint.angular_waypoint.append(angular_waypoint)
        #print(waypoint)
        trajectory.duration = 0
        trajectory.use_optimal_blending = False
        trajectory.waypoints.append(waypoint)
        
        # Validate before proceeding
        try:
            res = self.validate_waypoint_list(trajectory)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ValidateWaypointList")
            return False

        # Kinova's official validation process
        # NOTE: Kinova APIs check if physical joints reach limits
        # Critical for safe robot joint manipulation
        error_number = len(res.output.trajectory_error_report.trajectory_error_elements)

        while (error_number >= 1 and angular_duration != MAX_ANGULAR_DURATION) :
            angular_duration += 1
            trajectory.waypoints[0].oneof_type_of_waypoint.angular_waypoint[0].duration = angular_duration

            try:
                res = self.validate_waypoint_list(trajectory)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ValidateWaypointList")
                return False

            error_number = len(res.output.trajectory_error_report.trajectory_error_elements)
        #print(error_number, res)

        if (angular_duration == MAX_ANGULAR_DURATION) :
            # It should be possible to reach position within 30s
            # WaypointList is invalid (other error than angularWaypoint duration)
            rospy.loginfo("WaypointList is invalid")
            return False


        # Send the angles
        #print(trajectory)
        rospy.loginfo("Sending the joint angles...")
        req.input.oneof_action_parameters.execute_waypoint_list.append(trajectory)

        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ExecuteWaypointjectory")
            return False
        else:
            return self._wait_for_action_end_or_abort()

    def dh_mats(self, n=(0,7)):
        current_angles = np.concatenate(([0], (self.position[:7]), [0]))
        M = np.identity(4)
        for i in range(n[0], n[1] + 1):
            theta = DH_PARAMETERS[i][3] + current_angles[i]
            M_new = np.array([	[np.cos(theta), -np.cos(DH_PARAMETERS[i][0])*np.sin(theta),  np.sin(DH_PARAMETERS[i][0])*np.sin(theta), DH_PARAMETERS[i][1]*np.cos(theta)],
                            [np.sin(theta),  np.cos(DH_PARAMETERS[i][0])*np.cos(theta), -np.sin(DH_PARAMETERS[i][0])*np.cos(theta), DH_PARAMETERS[i][1]*np.sin(theta)],
                            [            0,                np.sin(DH_PARAMETERS[i][0]),                np.cos(DH_PARAMETERS[i][0]), DH_PARAMETERS[i][2]],
                            [0, 0, 0, 1]])
            M = M @ M_new
        return M

    def dh_mat_inv(self, n):
        current_angles = np.concatenate(([0], (self.position[:7]), [0]))
        theta = DH_PARAMETERS[n][3] + current_angles[n]
        R_T = np.array([[							  np.cos(theta), 							  np.sin(theta),  						   0],
                        [-np.cos(DH_PARAMETERS[n][0])*np.sin(theta),  np.cos(DH_PARAMETERS[n][0])*np.cos(theta), np.sin(DH_PARAMETERS[n][0])],
                        [ np.sin(DH_PARAMETERS[n][0])*np.sin(theta), -np.sin(DH_PARAMETERS[n][0])*np.cos(theta), np.cos(DH_PARAMETERS[n][0])]])
        
        T = -R_T @ np.array([[DH_PARAMETERS[n][1]*np.cos(theta)],[DH_PARAMETERS[n][1]*np.sin(theta)],[DH_PARAMETERS[n][2]]])
        M_new = np.concatenate((R_T, T), axis=1)
        M_new = np.concatenate((M_new, np.array([[0, 0, 0, 1]])), axis=0)
        return M_new

    def dh_mat(self, n):
        current_angles = np.concatenate(([0], (self.position[:7]), [0]))
        theta = DH_PARAMETERS[n][3] + current_angles[n]
        M = np.array([	[np.cos(theta), -np.cos(DH_PARAMETERS[n][0])*np.sin(theta),  np.sin(DH_PARAMETERS[n][0])*np.sin(theta), DH_PARAMETERS[n][1]*np.cos(theta)],
                        [np.sin(theta),  np.cos(DH_PARAMETERS[n][0])*np.cos(theta), -np.sin(DH_PARAMETERS[n][0])*np.cos(theta), DH_PARAMETERS[n][1]*np.sin(theta)],
                        [            0,                np.sin(DH_PARAMETERS[n][0]),                np.cos(DH_PARAMETERS[n][0]), 			  DH_PARAMETERS[n][2]],
                        [			 0,											 0,											 0,									1]])
        return M

    def v_mat_trans(self, jv, n=7):
        jv = np.concatenate((np.array(jv), [0], [0]))
        W = np.zeros((4, 4))
        for i in range(n+1):
            M = self.dh_mat(i)
            M_inv = self.dh_mat_inv(i)
            W = M_inv @ W @ M
            W[0][1] -= jv[i]
            W[1][0] += jv[i]
        return W

    def send_joint_speeds_command(self, th_ds):
        # print('sending_joint speeds')

        if self.movement_blocked:
            th_ds = np.zeros(7)

        if self.going_home:
            return False

        if np.linalg.norm(th_ds) > 1e-4:
            #print('just went home false')
            self.just_went_home = False
        temp_base = Base_JointSpeeds()
        for i, v in enumerate(th_ds):
            temp_vel = JointSpeed()
            temp_vel.joint_identifier = i
            temp_vel.value = v
            temp_base.joint_speeds.append(temp_vel)
        
        # send the angles
        if self.robot_name != "none":
            try:
                self.joint_velocity_pub.publish(temp_base)

            except rospy.ServiceException:
                rospy.logerr("Failed to publish send_joint_velocity")
                return False
            else:
                # time.sleep(0.5)
                return True
    
    def send_cartesian_velocity(self, axes, ref_frame=1):
		# CARTESIAN_REFERENCE_FRAME_UNSPECIFIED = 0,
		# CARTESIAN_REFERENCE_FRAME_MIXED = 1,
		# CARTESIAN_REFERENCE_FRAME_TOOL = 2,
		# CARTESIAN_REFERENCE_FRAME_BASE = 3,
        if np.array_equal(axes, self.prev_cv_cmd):
            return False
        
        # Declare max velocities 
        MAXV_TX = 0.05
        MAXV_TY = 0.05
        MAXV_TZ = 0.08
        MAXV_RX = 0.5
        MAXV_RY = 0.3
        MAXV_RZ = 0.5
        MAXV_GR = 0.3 # gripper

        self.prev_cv_cmd = axes

        cmd = TwistCommand()
        cmd.reference_frame = ref_frame
        cmd.duration = 0
        cmd.twist.linear_x = axes[0] * MAXV_TX
        cmd.twist.linear_y = axes[1] * MAXV_TY
        cmd.twist.linear_z = axes[2] * MAXV_TZ
        cmd.twist.angular_x = axes[3] * MAXV_RX
        cmd.twist.angular_y = axes[4] * MAXV_RY
        cmd.twist.angular_z = axes[5] * MAXV_RZ
        
        if self.robot_name != "none":
            try:
                self.cartesian_velocity_pub.publish(cmd)
                self.just_went_home = False
            except rospy.ServiceException:
                rospy.logerr("Failed to publish send_cartesian_velocity")
                return False
            else:
				# time.sleep(0.5)
                return True

    # #####
    # Gripper Control
    # #####
    def send_gripper_command(
        self, 
        value: float,
        ):
        try:
            assert self.is_gripper_present == True
        except:
            rospy.logerr("No gripper is present on the arm.")
            return False

        if value == self.prev_gripper_cmd:
            return
        
        self.prev_gripper_cmd = value
        # Initialize the request
        # Close the gripper
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_SPEED

        #rospy.loginfo("Sending the gripper command...")

        # Call the service 
        try:
            self.send_gripper_command_srv(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            #time.sleep(0.5)
            return True


    def __str__(self):
        string = "Kinova Gen3\n"
        string += "  robot_name: {}\n  dof: {}".format(self.robot_name, self.dof)
        return string
    

class RGBDVision(object):
    """Interact with any camera w/ topics published to ROS.
    Standard cv_bridge. 

    To run with eye-in-hand camera on Kinova Gen3, install ros_kortex_vision, then run:
        roslaunch kinova_vision kinova_vision.launch num_worker_threads:=0
    """
    def __init__(
        self,
        name: str,
        image_topic: str = "/camera/color/image_raw",
        depth_topic: str = "/camera/depth/image_raw",
        image_encoding: str = "bgr8",
        depth_encoding: str = "passthrough",
        ):
        self.name = name
        self.image_encoding = image_encoding
        self.depth_encoding = depth_encoding
        
        # Image frame & depth
        self.frame = None
        self.depth = None
        self.i = 0
        self.clicks = []
        self.enable_mouse_event = False 

        # initialize tracker 
        self.tracker = None 

        # Additional ops needed to run in align with image pipeline
        self.update_ops = {}

        # Subscriber
        try:
            self.bridge = CvBridge()
            self.image_sub = rospy.Subscriber(image_topic, Image, self._image_callback)
            self.depth_sub = rospy.Subscriber(depth_topic, Image, self._depth_callback)
        except:
            raise Exception("Failed to communicate with a camera!")

    def _image_callback(
        self,
        ros_image,
        ):
        """Handle raw image frame from Kinova Gen3 camera.
        BGR Frame: (480, 640, 3)
        """
        try:
            self.frame = self.bridge.imgmsg_to_cv2(ros_image, self.image_encoding)
        except CvBridgeError as e:
            print(e)
                       
        # Display
        vis = self.frame.copy()
        for (x, y) in self.clicks:
            cv2.circle(vis, (x, y), 2, (0, 0, 255), 2)

        # NOTE: Now run any additional update_ops
        data = {
            'vis': vis,
            'frame': self.frame,
        }
        try:
            for _, ops in self.update_ops.items():
                data = ops(data)
        except RuntimeError:    # Ignore badcallback situation
            pass

        cv2.imshow("{} RGB".format(self.name), data['vis'])
        if self.depth is not None: cv2.imshow("{} Depth".format(self.name), np.uint8(self.depth))
        key = cv2.waitKey(1)

        # NOTE: Remove later, simple save script
        if key == ord('s'):
            if self.frame is not None:
                cv2.imwrite("frame_{}.png".format(self.i), self.frame)
            if self.depth is not None:
                np.save("depth_{}.npy".format(self.i), self.depth)
            self.i += 1
            print("Image saved.")

        elif key == ord('c') and self.enable_mouse_event:   # ASCII dec: c
            cv2.setMouseCallback("{} RGB".format(self.name), self._mouse_event)

    def _depth_callback(
        self,
        ros_image,
        ):
        """Handle raw depth image from Kinova Gen3 camera.
        depth: (270, 480), uint16
        """
        try:
            # ros_kortex_vision publishes depth frame as 16UC1 encpding
            self.depth = self.bridge.imgmsg_to_cv2(ros_image, self.depth_encoding)
        except CvBridgeError as e:
            print(e)

        # Some image processing
        # TODO: Correctly normalize depth image frame
        # #####
        # #####
