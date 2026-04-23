import zmq
import json
import time

print("Starting Robot Dummy Node (ZMQ REP)...")

# 1. Setup ZMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555") # using * creates the server 

current_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

while True:
    print("\n[Robot] Waiting for command...")
    
    # 2. This line BLOCKS. The script pauses here until a message arrives.
    message_str = socket.recv_string()
    data = json.loads(message_str)
    
    print(f"[Robot] Received velocity command: {data['v_c']}")
    print("[Robot] Simulating arm movement...")
    
    # 3. Simulate the time it takes the arm to actually move
    time.sleep(1.5) 
    
    # Update our fake angles just so the data changes
    current_angles[0] += 0.1 
    
    # 4. Send the reply back to the vision node
    reply = {"status": "success", "q": current_angles}
    socket.send_string(json.dumps(reply))
    print(f"[Robot] Sent state reply: {reply['q']}")