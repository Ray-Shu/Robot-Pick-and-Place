# Pick-and-Place Robotics Project using Image Moments

This project was motivated through the idea of using image moments to dictate the movement of a Kinova Gen3 arm.

<p align="center">
  <img src="https://raw.githubusercontent.com/Ray-Shu/Robot-Pick-and-Place/assets/imgs/robot-setup1.png" width="430" alt="Robot Setup1" width="45%">
  <img src="https://raw.githubusercontent.com/Ray-Shu/Robot-Pick-and-Place/assets/imgs/robot-setup2.png" width="430" alt="Robot Setup2" width="45%">
  <br>
  <em>Setup views of the workspace environment</em>
</p>

# Project Description

I used calibrated visual servoing with an eye-in-hand setup, so that the camera velocity dictated the movement of the arm. 
The goal of my project was to have the robot arm pick and place shapes of different geometries and to test the effectiveness
of using image moments as the error signal for the control law:

$$\mathbf{v}_c = -\lambda \hat{L}_s^+ (\mathbf{s} - \mathbf{s}^*)$$

Where:
- $\mathbf{v}_c$ is the camera velocity command
- $\lambda$ is the positive control gain
- $\hat{L}_s^+$ is the pseudoinverse of the estimated interaction matrix
- $\mathbf{e} = \mathbf{s} - \mathbf{s}^*$ is the error between the target image moments, and the current image moments

For the feature vector $\mathbf{s}$, I used the first and second order image moments, which correspond to an objects centroid position, and its area:

$$\mathbf{s} = \begin{bmatrix} x & y & \log(a/a^*) \end{bmatrix}^{\top}$$

Where $x,y$ are normalized image coordinates, $a$ is the area, and $(a^*)$ is the target area. From experiments, I found that using $\log(a/a^\*)$ provided more stable movement 
over just using area.

The normalized image coordinates were computed from the camera intrinsics as

$$\begin{equation} x = \frac{c_x-u_0}{f}, \qquad y = \frac{c_y-v_0}{f}, \end{equation}$$

where $(u_0,v_0)$ is the principal point and $f$ is the focal length in pixels. I used a Logitech C922 PRO HD webcam for my experiments. 


# YOLO Detection and Segmentation 

To detect shapes and obtain their feature vectors, I used the yolo11n-seg model trained on a custom dataset of 3D geometric shapes. 
<p align="center">
  <img src="https://github.com/Ray-Shu/Robot-Pick-and-Place/blob/main/robot-classification/preview.jpg" width="430" alt="3d geometric shapes" width="80%">
  <br>
  <em>3D geometric shapes including: ellipse, cross, pentagon, cube, triangle, star, hexagon and circle.</em>
</p>

The model performed well on the dataset, as expected, reaching up to 0.97 mAP.
<p align="center">
  <img src="https://github.com/Ray-Shu/Robot-Pick-and-Place/blob/assets/imgs/yolo_mAP_inference.png" width="430" alt="yolo inference" width="80%">
</p>

Additionally, the model was able to consistently classify all shapes that it was trained on:
<p align="center">
  <img src="https://github.com/Ray-Shu/Robot-Pick-and-Place/blob/assets/imgs/cross_annotated.jpg" width="430" alt="yolo irl inference on a cross" width="80%">
</p>

# Communication between the Vision and the Robot node 

I hooked up a ZMQ server to connect the vision node (`robot-classification/vision_node.py`) and the robot node (`robot-visual-servoing/catkin_ws/src/kortex_bringup/vs_main.py`). The idea was that the robot node 
would solely be on the movement logic of the robot, and the vision node would be classifying shapes, calculating the feature vector, interaction matrix, and velocity updates. 

The robot node would be constantly pinging the the vision node for velocity commands, in which the vision node would return them. For this, I used ZMQ.REP and ZMQ.REQ to request and 
receive messages. 

---

For a comprehensive view of my project, feel free to check [my report](https://github.com/Ray-Shu/Robot-Pick-and-Place/blob/assets/pdfs/Robot_Pickup_and_Place.pdf), which 
details the algorithms I used (along with their pseudocode), additional formulas I've left out for the sake of brevity, and experiment results. 

# Citations 
1. François Chaumette. Image moments: a general and useful set of features for visual servoing. IEEE Transactions on Robotics, 2004, 20 (4), pp.713-723. ⟨inria-00352019⟩
