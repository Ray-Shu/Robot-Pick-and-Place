import libtmux
import time
from os import path
import sys

if __name__ == "__main__":
	server = libtmux.Server(
		config_file=path.expandvars("/home/user/robot-visual-servoing/scripts/.tmux.conf")
	)

	if server.has_session("sim"):
		exit()
	else:
		session = server.new_session("sim", start_directory="/home/user/robot-visual-servoing", attach=False)
		
	# terminals for the simulation to start
	terminals = {
		"kortex_bringup": "roslaunch kortex_bringup kortex_bringup.launch", # launch kortex - note that this starts a roscore
		"main": "sleep 1 && rosrun kortex_bringup vs_main.py", 
		"joy": "rosrun joy joy_node", 
		# "realsense_back": "sleep 5 && roslaunch realsense2_camera rs_camera.launch camera:=cam filters:=pointcloud depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30 align_depth:=true decimation_filter:=true spatial_filter:=true temporal_filter:=true hole_filling_filter:=true",
		# "rviz": "sleep 5 &&rviz -d /home/user/robot-visual-servoing/scripts/vs.rviz",
	}

	for name, cmd in terminals.items():
		window = session.new_window(name, attach=False)
		window.select_layout(layout="tiled")
		pane = window.panes[0]
		time.sleep(0.1)
		pane.send_keys(cmd, suppress_history=True)