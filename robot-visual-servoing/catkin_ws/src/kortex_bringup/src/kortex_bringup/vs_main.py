#!/usr/bin/python3

import rospy
import numpy as np
from kortex_driver.srv import *
from kortex_driver.msg import *
from sensor_msgs.msg import Joy, PointCloud2, JointState
from control_utils.ik_utils import png_control, cartesian_control, joint_control, xbox_control
from control_utils.kinova_gen3 import RGBDVision
import time
import math
import zmq
import json
import message_filters
from kinova_util import KinovaUtil
import sys
import ros_numpy
import msgpack

from std_msgs.msg import Int32MultiArray, Float32, Float64MultiArray, Int16
from kortex_driver.msg._BaseCyclic_Feedback import BaseCyclic_Feedback

import msgpack_numpy
msgpack_numpy.patch()
msgpack_numpy_encode = msgpack_numpy.encode
msgpack_numpy_decode = msgpack_numpy.decode

class CustomCommand():
    def __init__(self, ax, mode, trans_gain, rot_gain, wrist_gain):
        self.ax = ax
        self.mode = mode
        self.trans_gain = trans_gain
        self.rot_gain = rot_gain
        self.wrist_gain = wrist_gain

def gen_iris(base):
    class IrisRecord(base):
        def __init__(self):
            super(IrisRecord, self).__init__(None)
            self.ku = KinovaUtil()
            self.mode = 0 # modes for control
            self.prev_button_2 = 0 # prev button 2 to prevent double clicks
            self.prev_gripper_cmd = 0.0 # prev gripper cmd
            self.gripper_cmd = 0.0 # gripper cmd
            self.axes_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # joystick cmd

            ## open the gripper 
            self.send_gripper_command(0.2)
            rospy.sleep(1)
            self.send_gripper_command(0.0)

            self.home_array_rad = np.array([0.457, 0.376, 2.776, -1.077, 0.139, -1.584, -1.575])
            self.home_array = np.degrees(self.home_array_rad) % 360 # home array in deg
            # self.home_array = np.array([0.1, 65, -179.9, -120, 0, 100, -90]) # home array in deg
            # self.home_array = np.array([30, 14, 148, -80, 57, 3, -140]) # home array in deg
            self.send_joint_angles(self.home_array) #sends robot home

            self.predefined = False # a robot state 
            self.predefined_data = {} # predefined data required to move the robot 

            self.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL

            ### HARDCODED JOINT ANGLES FOR PERIMETER SWEEP 
            upper_left_position  = np.degrees(np.array([-0.164, 0.555, 2.749, -0.675, 0.293, -1.834, -2.109])) % 360
            upper_right_position = np.degrees(np.array([0.651, 0.707, 2.817, -0.379, 0.218, -1.941, -1.204]))  % 360
            lower_left_position  = np.degrees(np.array([-0.017, 0.305, 2.228, -1.158, 0.353, -1.697, -2.556])) % 360 
            lower_right_position = np.degrees(np.array([1.095, 0.176, 2.767, -1.224, 0.021, -1.633, -0.991]))  % 360
            
            self.perimeter_search = False 
            self.sweep_index = 0
            self.sweep_waypoints = [
                upper_left_position,
                upper_right_position,
                lower_right_position,
                lower_left_position,
            ]

            self.window_center = (424, 240)
            self.grip_center = None
            self.orientation = None
            self.infer = False
            self.stage = 1
            self.time_last = time.time()
            self.joy_type = 1
            self.auto = False
            self.agent_pos = None
            self.prev_robot_pos = None
            self.pc = None
            self.pointclouds = []
            self.states = []
            self.first = True
            self.trial_started = False
            self.trial_time = None
            self.reset_count = 0
            self.mode_switches = 0
            self.tooldata = [0, 0, 0, 0, 0, 0]

            self.moved = False

            self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)
            self.tool_sub = rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self.tool_callback)
            self.orient_sub = rospy.Subscriber("/my_gen3/inference", Float64MultiArray, self.orient_callback)
            self.cartesian_vel_pub = rospy.Publisher("/my_gen3/in/cartesian_velocity", TwistCommand, queue_size=10)

            # ZeroMQ setup
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://localhost:5555")
            self.socket.setsockopt(zmq.RCVTIMEO, -1) # hang for indefinite time  
            self.socket.setsockopt(zmq.LINGER, 0)       # Don't hang on close if server is dead

            # watchdog to ensure no stale velocities get inputted
            self.last_twist_time = time.time()
            self.watchdog_timeout = 0.7
            self.watchdog_timer = rospy.Timer(rospy.Duration(0.05), self.watchdog_cb)


            rospy.loginfo("Attempting to connect to visual node...")
            try:
                ping_msg = {"vision_state": "ping"}
                self.socket.send_json(ping_msg)
                
                response_msg = self.socket.recv_json()
                rospy.loginfo(response_msg)
                
                rospy.loginfo("✅ Connected to visual node")
            except zmq.Again:
                rospy.logerr("❌ Could not connect to visual node (timeout)")
                sys.exit(1)

            self.cam_sub = rospy.Subscriber("/cam/depth/color/points", PointCloud2, self.pc_callback)
            self.joint_sub = rospy.Subscriber("/my_gen3/joint_states", JointState, self.joint_callback)

            rospy.loginfo("🤖 Visual node initialized")
            
            self.run = True

            self.custom_commands = []
            print(f"Current mode = {self.mode}\a", end="\r")
            
        def mode_switch(self):
            self.mode = (self.mode + 1) % 2
            self.mode_switches += 1
            print(f"Current mode = {self.mode}\a", end="\r")
            return

        def joint_callback(self, joint_msg):
            self.agent_pos = list(joint_msg.position[:8])
            self.agent_pos[7] = 1 if self.agent_pos[7] > 0.7 else 0

        def pc_callback(self, pc_msg):
            self.pc = pc_msg

        def tool_callback(self, msg):
            rads = self.ku.get_eef_pose()[3:]
            self.tooldata = [
                msg.base.tool_pose_x, 
                msg.base.tool_pose_y, 
                msg.base.tool_pose_z, 
                msg.base.tool_pose_theta_x, 
                msg.base.tool_pose_theta_y,
                msg.base.tool_pose_theta_z
            ]


        def get_msg(self):
            self.socket.send_json({"hello": True})

            incoming = self.socket.recv_json()
            print(incoming)

        def orient_callback(self, msg):
            shape = [dim.size for dim in msg.layout.dim]
            temp = np.array(msg.data).reshape(shape)
            self.orientation = temp[0]
            # ABSOLUTE
            self.target = self.orientation
            # ABSOLUTE

            # DIFF
            # self.old_position = self.tooldata[3:]
            # self.target = self.old_position + self.orientation
            # DIFF

        def joy_callback(self, msg):
            self.buttons = msg.buttons
            
            if self.joy_type == 0:
                self.axes_vector = msg.axes
                MAXV_GR = 0.3

                # check for gripper commands
                if msg.buttons[0]: # trigger button - close gripper
                    self.gripper_cmd = -1 * MAXV_GR
                elif msg.buttons[1]: # button by thumb - open gripper
                    self.gripper_cmd = MAXV_GR
                else: # both buttons 0 and 1 are zero
                    self.gripper_cmd = 0.0

                if msg.buttons[2] == 0 and self.prev_button_2 == 1:
                    self.mode_switch()
                    
                self.prev_button_2 = msg.buttons[2]
                    
                if msg.buttons[3]:
                    if time.time() - self.time_last > 0.5:
                        self.infer = not self.infer
                        rospy.loginfo(f"Inference: {'True' if self.infer else 'False'}")
                        self.time_last = time.time()
                        self.mode_switches += 1

                if msg.buttons[4]:
                    pass

                if msg.buttons[5]:
                    pass

                if msg.buttons[6]:
                    pass

                if msg.buttons[7]:
                    pass

                if msg.buttons[8]:
                    # button 8 pressed, send robot home
                    if time.time() - self.time_last > 0.5:
                        self.run = False
                        rospy.loginfo(f"Trial State: PAUSED")
                        self.send_joint_speeds_command(np.zeros(7))
                        self.send_joint_angles(self.home_array)
                        rospy.loginfo("Button 8 pressed: sending robot to starting position")
                        self.reset_count += 1
                        self.run = True
                        self.time_last = time.time()
                        self.time_paused = time.time()

                if msg.buttons[9]:
                    if not self.trial_started  and time.time() - self.time_last > 0.5:
                        self.trial_started = True
                        self.trial_time = time.time()
                        self.reset_count = 0
                        self.mode_switches = 0
                        rospy.loginfo(f"Trial State: STARTED")
                        self.time_last = time.time()
                    elif self.trial_started  and time.time() - self.time_last > 0.5:
                        self.trial_started = False
                        rospy.loginfo(f"Trial Time: {time.time() - self.trial_time}")
                        rospy.loginfo(f"Trial State: SUCCESS")
                        rospy.loginfo(f"Trial Reset Count: {self.reset_count}")
                        rospy.loginfo(f"Trial Mode Switches: {self.mode_switches}")
                        self.time_last = time.time()

                if msg.buttons[10]:
                    if time.time() - self.time_last > 0.5:
                        rospy.loginfo(f"Trial State: RESUMED")
                        temp = time.time() - self.time_paused
                        self.trial_time += temp
                        self.time_paused = 0
                        self.time_last = time.time()

                if msg.buttons[11]:
                    if self.trial_started and time.time() - self.time_last > 0.5:
                        self.trial_started = False
                        rospy.loginfo(f"Trial Time: {180}")
                        rospy.loginfo(f"Trial State: FAIL")
                        rospy.loginfo(f"Trial Reset Count: {self.reset_count}")
                        rospy.loginfo(f"Trial Mode Switches: {self.mode_switches}")
                        self.time_last = time.time()


            elif self.joy_type == 1:
                MAXV_GR = 0.3
                roll = msg.buttons[1] - msg.buttons[3]
                self.axes_vector = [msg.axes[1], msg.axes[0], (1/(msg.axes[5]+1.1) - 1/(msg.axes[2]+1.1))/10, -msg.axes[4]/2, msg.axes[3], roll]

                if msg.buttons[0]:
                    pass

                if msg.buttons[1]:
                    pass

                if msg.buttons[2]:
                    pass
                    
                if msg.buttons[3]:
                    pass

                if msg.buttons[4]: # LB - open gripper
                    self.gripper_cmd = MAXV_GR

                elif msg.buttons[5]: # RB - close gripper
                    self.gripper_cmd = -1 * MAXV_GR

                else: # both buttons 0 and 1 are zero
                    self.gripper_cmd = 0.0

                if msg.buttons[6]:
                    # Start button pressed, send robot home
                    self.run = False
                    self.send_joint_speeds_command(np.zeros(7))
                    self.send_joint_angles(self.home_array)
                    rospy.loginfo("Start button pressed: sending robot to starting position")
                    self.run = True

                if msg.buttons[7]:
                    if time.time() - self.time_last > 0.5:
                        self.infer = not self.infer
                        rospy.loginfo(f"Inference: {'True' if self.infer else 'False'}")
                        self.time_last = time.time()
                
                if msg.buttons[8]:
                    pass

                if msg.buttons[9]:
                    pass

                if msg.buttons[10]:
                    pass

        def get_orientation(self):
            if self.orientation is not None:
                # rospy.loginfo(f"Target: {self.target}")
                # rospy.loginfo(f"Current: {self.tooldata[3:]}")
                diff = [abs(self.target[i] - self.tooldata[3+i]) for i in range(3)]
                # rospy.loginfo(f"Diff: {diff}")
                if self.joy_type == 0:
                    velocities = np.array([
                        (-1 if diff[2] > 180 else 1)*0.05 * ((self.target[2] - self.tooldata[5]) if abs(self.target[2] - self.tooldata[5]) > 0.5 else 0), #Correct axes placement
                        (-1 if diff[0] > 180 else 1)*0.05 * ((self.target[0] - self.tooldata[3]) if abs(self.target[0] - self.tooldata[3]) > 0.5 else 0), #Correct axes placement
                        (1 if diff[1] > 180 else -1)*0.05 * ((self.target[1] - self.tooldata[4]) if abs(self.target[1] - self.tooldata[4]) > 0.5 else 0), #Correct axes placement
                    ])
                if self.joy_type == 1:
                    velocities = np.array([
                        0,
                        0,
                        0,
                        (-1 if diff[0] > 180 else 1)*0.05 * ((self.target[0] - self.tooldata[3]) if abs(self.target[0] - self.tooldata[3]) > 0.5 else 0),
                        (1 if diff[2] > 180 else -1)*0.05 * ((self.target[2] - self.tooldata[5]) if abs(self.target[2] - self.tooldata[5]) > 0.5 else 0), #Correct axes placement,
                        (-1 if diff[1] > 180 else 1)*0.05 * ((self.target[1] - self.tooldata[4]) if abs(self.target[1] - self.tooldata[4]) > 0.5 else 0),
                    ])
                # rospy.loginfo(f"Resulting velocities: {velocities}")

                self.custom_commands.append(CustomCommand(velocities, 1, 1, 1, 1))

        def auto_pos(self):
            if self.pc is not None:
                MAXV_GR = 0.7
                if self.first:
                    self.pointclouds = [self.pc.astype(np.float16), self.pc.astype(np.float16)]
                    self.states = [np.zeros(8), np.zeros(8)]
                    self.prev_robot_pos = self.agent_pos
                if len(self.pointclouds) == self.n_obs:
                    start_time = time.time()
                    rospy.loginfo("Sending data to server")
                    payload = {
                        "agent_pos": np.array(self.states),
                        "point_cloud": np.array(self.pointclouds)
                    }
                    
                    # Send via ZeroMQ
                    self.socket.send(msgpack.packb(payload, default=msgpack_numpy_encode, use_bin_type=True))

                    try:
                        response = self.socket.recv()
                    except zmq.Again:
                        rospy.logwarn("Inference server timeout")
                        return

                    result = msgpack.unpackb(response, object_hook=msgpack_numpy_decode, raw=False)

                    # Extract and publish action
                    action = result["action"][0][:8]
                    actions = [[float(math.degrees(x)) for x in row[:7]]+[row[7]] for row in action]
                    # actions = actions[14:]
                    
                    for temp in actions:
                        print(temp)

                    self.pointclouds = []
                    self.states = []
                    self.run = False
                    for a in range(len(actions)):
                        self.send_joint_speeds_command(np.zeros(7))
                        rospy.sleep(0.1)
                        # actions[a][2] = 0
                        to_pos = [math.degrees(self.agent_pos[z]) + actions[a][z] for z in range(7)]
                        rospy.loginfo(f"AGENT POS: {self.agent_pos}")
                        rospy.loginfo(f"PREDICTION: {actions[a]}")
                        rospy.loginfo(f"TO_POS: {to_pos}")
                        self.send_joint_angles(np.array(to_pos))
                        rospy.loginfo("Sending joint commands")
                        rospy.sleep(0.5)

                        if a >= len(actions)-2:
                            self.pointclouds.append(self.pc)
                            current = self.agent_pos
                            diffs = [current[z] - self.prev_robot_pos[z] for z in range(7)] + [self.agent_pos[7]]
                            self.states.append(np.array(diffs, dtype=np.float16))
                        
                        self.prev_robot_pose = self.agent_pos

                        if actions[a][7] < -0.7:
                            success = self.send_gripper_command(MAXV_GR)
                        elif actions[a][7] > 0.7: 
                            success = self.send_gripper_command(-1 * MAXV_GR)
                        else:
                            success = self.send_gripper_command(0)
                    
                    self.run = True
                    
                    if self.first:
                        self.first = False

        def publish_stop(self): 
            stop_cmd = TwistCommand()
            stop_cmd.reference_frame = 1
            stop_cmd.duration = 0
            stop_cmd.twist.linear_x = 0.0
            stop_cmd.twist.linear_y = 0.0
            stop_cmd.twist.linear_z = 0.0
            stop_cmd.twist.angular_x = 0.0
            stop_cmd.twist.angular_y = 0.0
            stop_cmd.twist.angular_z = 0.0
            self.cartesian_vel_pub.publish(stop_cmd)

        def watchdog_cb(self, _event):
            if not self.infer:
                return

            if time.time() - self.last_twist_time > self.watchdog_timeout:
                self.publish_stop()


        def reconnect_vision_socket(self):
            try:
                self.socket.close()
            except Exception:
                pass

            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://localhost:5555")
            self.socket.setsockopt(zmq.RCVTIMEO, 500)
            self.socket.setsockopt(zmq.LINGER, 0)


        def request_vision(self, payload):
            '''
            handles zmq syncing 
            '''

            try:
                self.socket.send_json(payload)
            except zmq.ZMQError as e:
                rospy.logwarn(f"Vision send failed: {e}")
                self.publish_stop()
                self.reconnect_vision_socket()
                return None

            try:
                return self.socket.recv_json()
            except zmq.Again:
                rospy.logwarn("Vision recv timeout")
                self.publish_stop()
                self.reconnect_vision_socket()
                return None
            except zmq.ZMQError as e:
                rospy.logwarn(f"Vision recv failed: {e}")
                self.publish_stop()
                self.reconnect_vision_socket()
                return None



        def calculate_control_law(self):
            response = self.request_vision({"vision_state": "get_velocities"})
            if response is None:
                return

            if response.get("status") != "success":
                self.publish_stop()
                return

            # 3. If the vision node saw a target and sent velocities, use them!
            if response.get("status") == "success":
                
                if response.get("type") == "tracking": 
                    print("TRACKING")
                    v_c = response["velocities"]

                    map_vx = v_c["vx"]  
                    map_vy = v_c["vy"]  
                    map_vz = v_c["vz"]  # Z usually stays positive if pointing forward
                    
                    map_wx = v_c["wx"]
                    map_wy = v_c["wy"]
                    map_wz = v_c["wz"]
            
                    # Create the Kinova Twist Command
                    twist_cmd = TwistCommand()
                    twist_cmd.reference_frame = self.reference_frame 
                    twist_cmd.duration = 0        # 0 = keep moving until a new command is sent
                    
                    speed = 1

                    twist_cmd.twist.linear_x = map_vx / speed
                    twist_cmd.twist.linear_y = map_vy / speed
                    twist_cmd.twist.linear_z = map_vz / speed
                    twist_cmd.twist.angular_x = map_wx / speed
                    twist_cmd.twist.angular_y = map_wy / speed
                    twist_cmd.twist.angular_z = map_wz / speed

                    self.cartesian_vel_pub.publish(twist_cmd)
                    self.last_twist_time = time.time()

                    # --- DEBUG PRINT STATEMENTS ---
                    rospy.loginfo("--- Sent Twist Commands ---")
                    rospy.loginfo(f"Linear  [X, Y, Z] : [{twist_cmd.twist.linear_x:.4f}, {twist_cmd.twist.linear_y:.4f}, {twist_cmd.twist.linear_z:.4f}]")
                    rospy.loginfo(f"Angular [X, Y, Z] : [{twist_cmd.twist.angular_x:.4f}, {twist_cmd.twist.angular_y:.4f}, {twist_cmd.twist.angular_z:.4f}]")
                    rospy.loginfo("---------------------------")

                if response.get("type") == "aligned":
                    print("ALIGNING")
                    # save data 
                    self.predefined_data = {
                        "width_short_side": response.get("object_width_m"),
                        "move_duration": response.get("move_duration"),
                        "velocities": response.get("velocities"),
                        "object_span_px": response.get("object_span_px"),
                        "grasp_rect_angle_deg": response.get("grasp_rect_angle_deg"),
                    }

                    # switch from inference to predefined movement
                    self.publish_stop()
                    self.infer = False
                    self.predefined = True

            else:
                # If the status isn't "success" (e.g., target lost), stop the arm safely
                self.publish_stop()
            
        def run_predefined_sequence(self):
            """
            Runs a predefined sequence where the robot: 
            1. moves forward to align the grippers with the object
            2. closes the grippers to clasp the object
            3. returns to the home point (self.home_array) 
            """

            if not self.predefined_data:
                self.predefined = False
                return

            data = self.predefined_data
            width_short_side = data.get("width_short_side")
            move_duration = data.get("move_duration")
            v_c = data.get("velocities")

            if v_c is None:
                self.publish_stop()
                self.predefined_data = {}
                self.predefined = False
                return

            # MOVE FORWARD 
            twist_cmd = TwistCommand()
            twist_cmd.reference_frame = self.reference_frame
            twist_cmd.duration = 0  # keep moving until another command is sent

            twist_cmd.twist.linear_x = v_c["vx"]
            twist_cmd.twist.linear_y = v_c["vy"]
            twist_cmd.twist.linear_z = v_c["vz"]
            twist_cmd.twist.angular_x = v_c["wx"]
            twist_cmd.twist.angular_y = v_c["wy"]
            twist_cmd.twist.angular_z = v_c["wz"]

            rate = rospy.Rate(30)  # 30 Hz = 30 loops per second
            
            end_time =  rospy.Time.now() + rospy.Duration(move_duration)
            while rospy.Time.now() < end_time and not rospy.is_shutdown():
                self.cartesian_vel_pub.publish(twist_cmd)
                self.last_twist_time = time.time()
                rate.sleep()

            self.publish_stop()

            # CLASP OBJECT
            k = 1.5
            if width_short_side is None:
                grasp_close_time = 0.5
            else:
                grasp_close_time = float(np.clip(k * width_short_side, 0.25, 0.8))
            print("GRASP CLOSE TIME: ", grasp_close_time)

            rospy.sleep(0.2)

            self.send_gripper_command(-0.3)
            rospy.sleep(grasp_close_time)

            self.send_gripper_command(0.0)
            rospy.sleep(0.2)

            self.send_joint_angles(self.home_array)

            self.predefined_data = {}
            self.predefined = False
            self.infer = True
            self.perimeter_search = True 


        def query_vision_for_bowl(self):
            """
            sends a message to the vision node if the bowl is sighted 
            """
            response = self.request_vision({"vision_state": "search_bowl"})

            if response is None:
                return {"found": False}

            return {
                "found": bool(
                    response.get("status") == "success" and
                    response.get("found_bowl", False)
                )
            }
        

        def run_search_step(self): 
            """
            Runs a perimeter sweep across hardcoded angles:
            """ 

            # 1. Ask vision if bowl is visible before moving
            detection = self.query_vision_for_bowl()

            if detection["found"]:
                self.publish_stop()
                self.perimeter_search = False 
                return

            # 2. Move to next sweep waypoint
            next_joints = self.sweep_waypoints[self.sweep_index]
            self.send_joint_angles(np.asarray(next_joints))

            # 3. Hold briefly and check again
            rospy.sleep(0.3)

            detection = self.query_vision_for_bowl()
            if detection["found"]:
                self.publish_stop()
                self.perimeter_search = False 
                return

            # 4. Advance waypoint index
            self.sweep_index = (self.sweep_index + 1) % len(self.sweep_waypoints)


        def step(self):
            if self.run:
                self.custom_commands = []

                # do visual servoing 
                if self.infer:
                    if self.perimeter_search: # do a perimeter search across hardcoded joint angles  
                        self.run_search_step()  
                    else: 
                        self.calculate_control_law()
                
                elif self.predefined: 
                    self.run_predefined_sequence() 

                else:
                    # step according to rospy rate
                    if self.joy_type == 0:
                        super().step(self.axes_vector, self.mode, self.custom_commands)
                    elif self.joy_type == 1:
                        super().step(self.axes_vector, self.custom_commands)
                    if self.gripper_cmd != self.prev_gripper_cmd:
                        success = self.send_gripper_command(self.gripper_cmd)
                        self.prev_gripper_cmd = self.gripper_cmd

    return IrisRecord()

def main():
    controller = xbox_control# can replace with png_control, cartesian_control or joint_control or xbox_control
    robot = gen_iris(controller)
    
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        robot.step()
        rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node('iris_control', anonymous=True)
        main()
    except rospy.ROSInterruptException:
        print("ROSInterruptException")
