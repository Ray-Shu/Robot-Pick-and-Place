#!/usr/bin/python3

import numpy as np
from control_utils.kinova_gen3 import KinovaGen3

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

def rot_x(theta):
	mat = [[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0], [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]]
	return np.array(mat)

def rot_y(theta):
	mat = [[np.cos(theta), 0, np.sin(theta), 0], [0, 1, 0, 0], [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]]
	return np.array(mat)

def rot_z(theta):
	mat = [[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
	return np.array(mat)

def rot_2d(theta):
	mat = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
	return np.array(mat)

class png_control(KinovaGen3):
    def __init__(self, args):
        super(png_control, self).__init__()
        self.run = True # prevents robot from stepping
        self.ee_flat = np.zeros(7) # placeholder vector to operate last joint to keep it flat
        self.lengths = np.array([156.4, 128.4, 210.4, 210.4, 208.4, 105.9, 105.9, 61.5, 140]) # lengths of robot joints, could replace with dh-params
        self.last_j6_error = 0 # for pid controller to keep ee flat
        self.j6_angle = 0 # to rotate inputs into more intuitive reference frame
        self.rotation_gain = 0.2 # gains for rotations
        self.translation_gain = 0.2 # gains for translations
        self.wrist_motion_gain = 0.2 # gains for wrist flexion

    def step(self, ax, mode):
        # runs each loop according to rospy.rate
        if self.run:
            cmd = self.remap_axes(ax, mode)
            self.send_joint_speeds_command(cmd)

    def remap_axes(self, vec, mode):
        # multiplies joystick commands, vec, with qdots
        if mode == 0:
            twist = self.threshold(vec[5], 0.5) #thresholding third joystick dof, twist is vec[2] but currently mapped to vec[5] (thumb button)
        else:
            twist = self.threshold(vec[4], 0.5)
        control_vec = np.array([[1, vec[1], vec[0], twist]]) # defining control vec, can change indicies of vec to use different dofs of joystick
        self.state_to_qdot(mode) # updating qdots
        if mode == 1:
             self.j6_theta_change(-twist) # this is to change the parameter of the pid controller operating the last joint
        # finding the final actions sent to the robot
        cmd = np.dot(control_vec, np.array([self.ee_flat, self.fb_qdot, self.lr_qdot, self.tw_qdot]))[0].tolist()
        cmd = self.check_cmd(cmd) # making sure we won't hit any joint limits
        return cmd # send the commands to robot
    
    def state_to_qdot(self, mode):
        # this function defines bases for movements and converts them to joint velocity commands
        # bases are in the form of arrays where [x, y, z, a, b, 0, i, j]
        # x y z are translations in base coordinates
        # a b are rotations in end-effector coordinates
        # i j  are wrist rotations in end-effector coordinates
        # the 0 is a placeholder as we usually would define it to operate the last joint of the end-effector but we use a pid controller instead

        ee_z = self.get_ee_z_frame() # getting end effector coordinate frame in base coordinates

        # translation mode
        if mode == 0:
            fb_basis = np.array([ee_z[0][0], ee_z[1][0], 0, 0, 0, 0, 0, 0]) # translations defined by first dof of joystick
            lr_basis = np.array([0, 0, 0, 0, 0, 0, -np.cos(self.j6_angle), -np.sin(self.j6_angle)]) # translations defined by second dof of joystick
            tw_basis = np.array([0, 0, 1, 0, 0, 0, 0, 0]) # translations defined by third dof of joystick
            self.fb_qdot, self.lr_qdot, self.tw_qdot = self.parse_bases(fb_basis, lr_basis, tw_basis) # getting qdots

        # rotation mode
        if mode == 1:
            fb_basis = np.array([0, 0, 0, -np.sin(self.j6_angle), np.cos(self.j6_angle), 0, 0, 0])
            lr_basis = np.array([0, 0, 0, -np.cos(self.j6_angle), -np.sin(self.j6_angle), 0, 0, 0])
            tw_basis = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            self.fb_qdot, self.lr_qdot, self.tw_qdot = self.parse_bases(fb_basis, lr_basis, tw_basis)
    
    def get_ee_z_frame(self):
        # converts the end-effector frame to basis coordinates using dh_parameters
        ref_x_ee = np.array([[1], [0], [0], [1]])
        ref_z_ee = np.array([[0], [0], [1], [1]])
        origin_pt = np.array([[0], [0], [0], [1]])
        T_base_ee = self.dh_mats((0,7))
        ref_z_base = T_base_ee @ ref_z_ee
        ee_origin_base = T_base_ee @ origin_pt
        ray_full_base = ref_z_base - ee_origin_base
        delta_z = (T_base_ee @ (ref_x_ee.T @ rot_z(self.j6_angle)).T)[2] - ee_origin_base[2]
        self.j6_control(-delta_z)
        return ray_full_base

    def parse_bases(self, fb, lr, tw):
        # takes in movement bases and calls the corrensponding methods
        bases = (fb, lr, tw)
        qdots = []
        for basis in bases:
            qdot1 = np.zeros(7)
            qdot2 = np.zeros(7)
            qdot3 = np.zeros(7)
            if np.any(basis[0:3]):
                qdot1 = self.translate_qdot(basis)*self.translation_gain
            if np.any(basis[3:5]):
                qdot2 = self.rotate_qdot(basis)*self.rotation_gain
            if np.any(basis[6:8]):
                qdot3 = self.wrist_qdot(basis)*self.wrist_motion_gain
            qdots.append(np.sum([qdot1, qdot2, qdot3], axis=0))
        return qdots

    def check_cmd(self, command):
        # making sure we won't hit any joint limits
        future_joint_pos = np.array(command)/40 + self.position[0:7]
        for i in range(7):
            if future_joint_pos[i] < JOINT_LIMIT[i][0] or future_joint_pos[i] > JOINT_LIMIT[i][1]:
                print(f"COMMAND ERROR - JOINT {i} LIMIT REACHED")
                return [0, 0, 0, 0, 0, 0, 0]
        return command

    def translate_qdot(self, basis):
        len = self.lengths

        # calculating basis vector translations
        base_vels = self.jacob_base_inv() @ basis[0:3]

        # finding excess rotations from these base vel translations
        jv_base = np.array([base_vels[0], base_vels[1], 0, base_vels[2], 0, 0, 0])
        W = self.v_mat_trans(n=7, jv=jv_base)
        base_vels_ee = [W[2][1], W[0][2]]

        # finding corrections for base joint velocities to maintain ee in ray direction
        lat_goal = np.array([base_vels_ee[1], base_vels_ee[0]]) * (len[6] + len[7])
        lat_joint_vel_2D = self.jacob_ee_inv() @ lat_goal

        # combining base joint and ee joint velocities
        reach_vector_6 = np.array([base_vels[0], base_vels[1], 0, base_vels[2], lat_joint_vel_2D[0], lat_joint_vel_2D[1], 0])
        W = self.v_mat_trans(n=7, jv=reach_vector_6)
        reach_vector = np.array([base_vels[0], base_vels[1], 0, base_vels[2], lat_joint_vel_2D[0], lat_joint_vel_2D[1], -W[0][1]])
        return reach_vector/np.linalg.norm(reach_vector)

    def rotate_qdot(self, basis):
        # finding relevant coordinate frames
        origin_pt = np.array([[0], [0], [0], [1]])
        dh_limit = 8
        T_base_ee = self.dh_mats((0,dh_limit))
        ee_origin_base = T_base_ee @ origin_pt
        
        # finding rotations of wrist
        lat_goal = [basis[3], basis[4]]
        lat_joint_vel_2D = self.jacob_ee_inv() @ lat_goal
        jv_base_sph_lat = [0, 0, 0, 0, lat_joint_vel_2D[0], lat_joint_vel_2D[1], 0]
        W_lat = self.v_mat_trans(n=dh_limit, jv=jv_base_sph_lat)
        v_lat = np.array([[-W_lat[0][3]*1000], [-W_lat[1][3]*1000], [-W_lat[2][3]*1000], [1]]) # finding excess translations

        # offsetting ee translations with base velocities       
        v_lat_ee_base = T_base_ee @ v_lat
        v_lat_ee_base_shifted = v_lat_ee_base - ee_origin_base
        base_vels_lat = self.jacob_base_inv() @ v_lat_ee_base_shifted[0:3]

        # offsetting ee translations with wrist velocities
        lat_vector = np.array([base_vels_lat[0][0], base_vels_lat[1][0], 0, base_vels_lat[2][0], lat_joint_vel_2D[0], lat_joint_vel_2D[1], 0])
        W_lat = self.v_mat_trans(n=dh_limit, jv=lat_vector)
        lat_goal = np.array([W_lat[0][3], -W_lat[1][3]])*1000
        lat_joint_vel_2D_2 = self.jacob_ee_inv(sphere=140) @ lat_goal
        lat_vector = np.array([base_vels_lat[0][0], base_vels_lat[1][0], 0, base_vels_lat[2][0], lat_joint_vel_2D[0] + lat_joint_vel_2D_2[0], lat_joint_vel_2D[1] + lat_joint_vel_2D_2[1], -W_lat[0][1]])

        return -lat_vector/np.linalg.norm(lat_vector)

    def wrist_qdot(self, goal):
        # wrist motions
        lat_goal = [goal[6], goal[7]]
        lat_joint_vel_2D = self.jacob_ee_inv() @ lat_goal
        lat_joint_vel_3D_6 = [0, 0, 0, 0, lat_joint_vel_2D[0], lat_joint_vel_2D[1], 0]
        W = self.v_mat_trans(n=7, jv=lat_joint_vel_3D_6)
        lat_joint_vel_3D = [0, 0, 0, 0, lat_joint_vel_2D[0], lat_joint_vel_2D[1], -W[0][1]]

        return lat_joint_vel_3D/np.linalg.norm(lat_joint_vel_3D)
    
    
    def threshold(self, x, t):
        # thresholding a control vector
        if np.abs(x) < t:
            x = 0
        else:
            x = (x - np.sign(x) * t)/(1-t)
        return x

    def j6_theta_change(self, ctrl):
        # changing pid controller params
        self.j6_angle += ctrl/100
        return

    def jacob_base_inv(self):
        # base frame ik
        angles = self.position
        l = self.lengths
        t0 = angles[0]
        t1 = angles[1]
        t3 = angles[3]
        phi = t1 - t3
        l1 = l[2] + l[3]
        l2 = l[4] + l[5]
        dxdt0 = -np.sin(t0)*(l1*np.sin(t1) + l2*np.sin(phi))
        dxdt1 = np.cos(t0)*(l1*np.cos(t1) + l2*np.cos(phi))
        dxdt3 = -np.cos(t0)*(l1*np.cos(phi))

        dydt0 = -np.cos(t0)*(l1*np.sin(t1) + l2*np.sin(phi))
        dydt1 = -np.sin(t0)*(l1*np.cos(t1) + l2*np.cos(phi))
        dydt3 = np.sin(t0)*(l1*np.cos(phi))

        dzdt0 = 0
        dzdt1 = -(l1*np.sin(t1)+l2*np.sin(phi))
        dzdt3 = l2*np.sin(phi)

        jacob_inv = np.linalg.inv(np.array([[dxdt0, dxdt1, dxdt3], [dydt0, dydt1, dydt3], [dzdt0, dzdt1, dzdt3]]))
        
        return jacob_inv

    def jacob_ee_inv(self, sphere=0):
        # wrist frame ik
        angles = self.position
        theta_2 = angles[5]
        theta_3 = angles[6]
        l = self.lengths[6] + self.lengths[7] + sphere

        a = -l * np.sin(theta_2) * np.sin(theta_3)
        b = -l * np.cos(theta_3)
        c = -l * np.sin(theta_2) * np.cos(theta_3)
        d = l * np.sin(theta_3) 

        jacob_inv = np.array([[d, -b], [-c, a]]) / (a * d - b * c)

        return jacob_inv

    def j6_control(self, in_p):
        # pid controller for joint 7 to keep it flat
        p = -in_p[0]
        kp = 3
        kd = 1
        d = p - self.last_j6_error
        self.last_j6_error = p
        self.ee_flat = np.array([0, 0, 0, 0, 0, 0, kp * p + kd * d])
        return

class cartesian_control(KinovaGen3):
    def __init__(self, args):
        # global 
        super(cartesian_control, self).__init__()
        self.run = True
        self.DOF = 3
        self.prev_cv_cmd = np.zeros(6)
        self.joint_velocities = np.zeros(6)
        self.home_array = np.array([0.1, 65, -179.9, -120, 0, 100, 90])
        self.z_dim = 6

    def step(self, ax, mode):
        if self.run:
            cmd = self.remap_axes(ax, mode)
            self.joint_velocities = cmd
            # if cmd != self.prev_cv_cmd: 
            self.send_cartesian_velocity(cmd)
            # self.prev_cv_cmd = cmd
            
    def remap_axes(self, vec, mode):
        if self.DOF == 3:
            if mode == 0:	
                cmd = [vec[1], vec[0], -0.05*vec[1] + vec[2], 0, 0, 0]
            elif mode == 1:
                cmd = [0, 0, 0, vec[1], -1.0 * vec[0], -1.0 * vec[2]]
                
        if self.DOF == 2:
            if mode == 0:	
                cmd = [vec[1], vec[0], 0, 0, 0, 0]
            elif mode == 1:
                cmd = [0, 0, vec[1], -1.0 * vec[0], 0, 0]
            elif mode == 2:
                cmd = [0, 0, 0, 0, vec[1], vec[0]]

        return np.array(cmd)      

class joint_control(KinovaGen3):
    def __init__(self, args):
        # global 
        super(joint_control, self).__init__()
        self.run = True
        self.joint_velocities = np.zeros(7)
        self.home_array = np.array([0.1, 65, -179.9, -120, 0, 100, 90])
        self.z_dim = 8
        self.control_scale = 1/4

    def step(self, ax, mode):
        if self.run:
            cmd = np.array(self.remap_axes(ax, mode))*self.control_scale
            self.joint_velocities = cmd
            self.send_joint_speeds_command(cmd)

    def remap_axes(self, vec, mode):
        if mode == 0:	
            cmd = [vec[1], vec[0], 0, 0, 0, 0, 0]
        elif mode == 1:
            cmd = [0, 0, vec[1], vec[0], 0, 0, 0]
        elif mode == 2:
            cmd = [0, 0, 0, 0, vec[1], vec[0], 0]
        elif mode == 3:
            cmd = [0, 0, 0, 0, 0, 0, vec[1]]
        return cmd  