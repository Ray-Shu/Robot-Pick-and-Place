import zmq
import time
import numpy as np

# ----------------------------------------------------------
# Setup ZMQ Client
# ----------------------------------------------------------
context = zmq.Context()
socket = context.socket(zmq.REQ)
print("Connecting to Visual Servoing server...")
socket.connect("tcp://localhost:5555")

# ----------------------------------------------------------
# Simulation Parameters
# ----------------------------------------------------------
# Initial simulated camera pose [x, y, z, roll, pitch, yaw]
pose = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) 

# Time step for integration (Hz)
HZ = 30.0
dt = 1.0 / HZ

print("Starting simulation loop. Press Ctrl+C to stop.")

try:
    # Initial ping to establish connection
    socket.send_json({"ping": True})
    response = socket.recv_json()
    if "pong" in response:
        print("Successfully connected to server.")

    while True:
        start_time = time.time()

        # 1. Request latest velocity from the server
        socket.send_json({"request": "get_velocity"})
        msg = socket.recv_json()

        # 2. Check if the server sent valid velocity data
        if "no data" in msg:
            print("[Status] No target detected. Holding position.")
        elif "vx" in msg:
            # Extract the velocity vector: v_c = [vx, vy, vz, wx, wy, wz]
            v_c = np.array([
                msg["vx"], 
                msg["vy"], 
                msg["vz"], 
                msg["wx"], 
                msg["wy"], 
                msg["wz"]
            ])

            # 3. Integrate velocities to update the simulated pose
            # Pose_new = Pose_old + Velocity * dt
            pose += v_c * dt

            # 4. Output the simulated state
            print(f"--- SIMULATED POSE UPDATE ---")
            print(f"Velocity rx:  [{v_c[0]:.4f}, {v_c[1]:.4f}, {v_c[2]:.4f}, {v_c[3]:.4f}, {v_c[4]:.4f}, {v_c[5]:.4f}]")
            print(f"New Pose:     [X: {pose[0]:.4f}, Y: {pose[1]:.4f}, Z: {pose[2]:.4f}, Rx: {pose[3]:.4f}, Ry: {pose[4]:.4f}, Rz: {pose[5]:.4f}]\n")

        # Regulate the simulation loop rate
        elapsed = time.time() - start_time
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
finally:
    socket.close()
    context.term()