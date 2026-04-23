import zmq
import json
import time

print("Starting Vision Dummy Node (ZMQ REQ)...")

# 1. Setup ZMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555") # connects to the server where robot is

for i in range(5): # Run 5 fake frames
    print(f"\n--- Vision Frame {i+1} ---")
    
    # Simulate YOLO/Moments processing time
    time.sleep(0.5) 
    fake_velocity = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # 2. Send the command
    command = {"command": "move", "v_c": fake_velocity}
    print(f"[Vision] Sending command: {command['v_c']}")
    socket.send_string(json.dumps(command))
    
    print("[Vision] Blocked. Waiting for robot to complete movement...")
    
    # 3. This line BLOCKS. We cannot process the next camera frame until the robot replies.
    reply_str = socket.recv_string()
    reply_data = json.loads(reply_str)
    
    print(f"[Vision] Received new robot state: {reply_data['q']}")

print("\n[Vision] Test complete.")