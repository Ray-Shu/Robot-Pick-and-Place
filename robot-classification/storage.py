import cv2
from ultralytics import YOLO
import numpy as np
import math
import json
import os
import zmq
import threading
import time 


model = YOLO('runs/segment/train/weights/best.pt')
CONFIDENCE_THRESHOLD = 0.70
OUTPUT = "ground_truth/"

os.makedirs(OUTPUT, exist_ok=True)
LAMBDA_GAIN = 0.01
AREA_EPSILON = 1e-6
VZ_MIN_MAGNITUDE = 0.003
VZ_MAX_MAGNITUDE = 0.025
VZ_DEADBAND = 0.001


class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()

        self.running = True
        
        # 1. Create the Lock
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True 
        self.thread.start()

    def update(self):
        while self.running:
            # Read from the camera (this takes time, so we do it OUTSIDE the lock)
            ret, frame = self.cap.read()
            print(ret, frame is None)
            
            # 2. Only update the variables if the frame is actually valid
            if ret:
                # Lock the door, update the variables, unlock the door
                with self.lock:
                    self.ret = ret
                    self.frame = frame

    def read(self):
        # Lock the door, copy the frame, unlock the door
        with self.lock:
            # 3. The 'None' guard: prevent crashing if the frame is empty
            if self.frame is not None:
                return self.ret, self.frame.copy()
            else:
                return False, None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# ----------------------------------------------------------
# Setup ZMQ Server
# ----------------------------------------------------------
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# ----------------------------------------------------------
# Control Law
# ----------------------------------------------------------
def control_law(latest_s, gt_features, is_symmetric):
    cx, cy, area, alpha = latest_s
    gt_area = gt_features["area"] # Get the target area

    # 1. Normalize BOTH current and ground truth coordinates
    x, y = normalize_coords(cx, cy)
    gt_x, gt_y = normalize_coords(gt_features["centroid_x"], gt_features["centroid_y"])

    # NORMALIZED AREA: Current area divided by Target Area
    #a_norm = area / gt_area
    #gt_a_norm = 1.0

    # 2. Compute error using the normalized coordinates!
    area_ratio = max(area, AREA_EPSILON) / max(gt_area, AREA_EPSILON)
    e = [
        x - gt_x,
        y - gt_y,
        math.log(area_ratio)
    ]

    if not is_symmetric and "alpha_degrees" in gt_features:
        e_alpha = math.degrees(alpha) - gt_features["alpha_degrees"]
        e_alpha = (e_alpha + 180) % 360 - 180
        e.append(math.radians(e_alpha)) 

    e = np.array(e)

    # Pass gt_area into the matrix calculator
    L = compute_interaction_matrix(latest_s, gt_area, is_symmetric) 
    L_pinv = np.linalg.pinv(L) 
    #print(L_pinv.shape, e.shape)
    v = -LAMBDA_GAIN * L_pinv @ e
    v[2] = shape_vertical_velocity(v[2])

    return v

# ----------------------------------------------------------
# Interaction Matrix and Camera Parameter Calculations
# ----------------------------------------------------------

# using C920 Pro HD Logitech Webcam
# focal length = 3.67mm 
# sensor width = 4.6mm
# focal length in px = f_mm * W_px / W_mm

FOCAL_LENGTH = 510.6
principal_x = 320
principal_y = 240  

# intrinsic matrix
K = [[FOCAL_LENGTH, 0, principal_x], 
     [0, FOCAL_LENGTH, principal_y],
     [0,            0,           1]]

# extrinsic matrix 
RT = [[1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0]]

def normalize_coords(cx, cy):
    x = (cx - principal_x) / FOCAL_LENGTH
    y = (cy - principal_y) / FOCAL_LENGTH
    return x, y

def compute_interaction_matrix(s, gt_area, is_symmetric, Z=1.0):
    # first run under the assumption that image plane is parallel to workshop plane 
    # this means that A = B = C = 0 

    cx, cy, area, alpha = s
    x,y = normalize_coords(cx, cy)

    L_xg = np.array([-1/Z, 0,    x/Z,             x*y,      -(1 + x**2), y])
    L_yg = np.array([0,   -1/Z,  y/Z,             1 + y**2, -x*y,       -x])
    L_area = np.array([0, 0, area*3/Z - area, 3*area*y, -3*area*x, 0])
    L_a = L_area / max(area, AREA_EPSILON)

    if not is_symmetric:
        row4 = np.array([0, 0, 0, 0, 0, -1])
        L = np.vstack([L_xg, L_yg, L_a, row4])
    else: 
        L = np.vstack([L_xg, L_yg, L_a])

    #print(L)

    return L

def shape_vertical_velocity(vz):
    abs_vz = abs(vz)
    if abs_vz < VZ_DEADBAND:
        return 0.0

    limited_vz = min(abs_vz, VZ_MAX_MAGNITUDE)
    boosted_vz = max(limited_vz, VZ_MIN_MAGNITUDE)
    return math.copysign(boosted_vz, vz)


# ---------------------------------------------------------
# Globals for Temporal Smoothing & Ground Truth Caching
# ---------------------------------------------------------
smooth_mu20, smooth_mu02, smooth_mu11 = None, None, None
alpha_smooth = 0.15 
gt_cache = {} # Caches loaded JSON files so we don't read the hard drive every frame

def reset_orientation_smoothing():
    global smooth_mu20, smooth_mu02, smooth_mu11
    smooth_mu20 = smooth_mu02 = smooth_mu11 = None

def load_ground_truth(class_name):
    '''Safely loads target features, returning None if missing or corrupted.'''
    filepath = os.path.join(OUTPUT, f"{class_name.lower()}_features.json")
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read {filepath}: {e}")
            return None # Failsafe: Pretend it doesn't exist
            
    return None # Failsafe: File actually doesn't exist

def save_ground_truth(frame, s, is_symmetric, class_name): 
    '''Saves the target features dynamically based on the class name.'''
    filepath = os.path.join(OUTPUT, f"{class_name.lower()}_features.json")
    img_path = os.path.join(OUTPUT, f"{class_name.lower()}_annotated.jpg")

    [cx, cy, area, alpha_rad] = s
    features = {"area": area, "centroid_x": cx, "centroid_y": cy}

    if not is_symmetric and alpha_rad is not None:
        features["alpha_degrees"] = math.degrees(alpha_rad)

    cv2.imwrite(img_path, frame)
    with open(filepath, "w") as f:
        json.dump(features, f, indent=4)
        
    print(f"\n[SAVED] Ground truth for {class_name} saved to {filepath}")
    
    # Update the cache immediately so the error display pops up
    gt_cache[class_name] = features 

def visualize_img_moments(frame, s, is_symmetric):
    '''Outputs current image moments numbers on the screen (Top Left).'''
    [cx, cy, area, alpha_in_rads] = s
    cv2.circle(frame, (int(cx), int(cy)), 7, (0, 0, 255), -1)

    cv2.putText(frame, f"Area: {int(area)} px", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Centroid: {int(cx)}, {int(cy)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if not is_symmetric and alpha_in_rads is not None: 
        line_len = 75
        end_x = int(cx + line_len * math.cos(alpha_in_rads))
        end_y = int(cy + line_len * math.sin(alpha_in_rads))
        cv2.line(frame, (int(cx), int(cy)), (end_x, end_y), (255, 0, 0), 3)
        cv2.putText(frame, f"Angle: {math.degrees(alpha_in_rads):.1f} deg", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def visualize_error(frame, s, gt_features, is_symmetric):
    '''Calculates e = s - s* and displays it on the screen (Top Right).'''
    [cx, cy, area, alpha_rad] = s
    
    # Calculate basic errors
    e_x = cx - gt_features["centroid_x"]
    e_y = cy - gt_features["centroid_y"]
    e_area = area - gt_features["area"]

    # Display on the right side of the screen
    start_x = 400
    cv2.putText(frame, f"ERROR (s - s*)", (start_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"e_x: {e_x:.1f} px", (start_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"e_y: {e_y:.1f} px", (start_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"e_area: {e_area:.1f}", (start_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"curr_area: {area:.1f}", (start_x, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    # Angle error requires special math to prevent 360-degree wrap-around bugs
    if not is_symmetric and alpha_rad is not None and "alpha_degrees" in gt_features:
        e_alpha = math.degrees(alpha_rad) - gt_features["alpha_degrees"]
        # Normalize the error to be between -180 and 180
        e_alpha = (e_alpha + 180) % 360 - 180 
        cv2.putText(frame, f"e_ang: {e_alpha:.1f} deg", (start_x, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def get_img_moments(binary_mask, is_symmetric, frame):
    global smooth_mu20, smooth_mu02, smooth_mu11
    
    M = cv2.moments(binary_mask, binaryImage=True)
    if M["m00"] != 0: 
        area = M["m00"]
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        alpha_in_rads = None
        if not is_symmetric:
            if smooth_mu20 is None:
                smooth_mu20, smooth_mu02, smooth_mu11 = M["mu20"], M["mu02"], M["mu11"]
            else:
                smooth_mu20 = alpha_smooth * M["mu20"] + (1 - alpha_smooth) * smooth_mu20
                smooth_mu02 = alpha_smooth * M["mu02"] + (1 - alpha_smooth) * smooth_mu02
                smooth_mu11 = alpha_smooth * M["mu11"] + (1 - alpha_smooth) * smooth_mu11

            alpha_in_rads = 0.5 * math.atan2(2 * smooth_mu11, smooth_mu20 - smooth_mu02)

        s = [cx, cy, area, alpha_in_rads]
        visualize_img_moments(frame, s, is_symmetric) 
        return s
    return None

# ---------------------------------------------------------
# Main Loop
# ---------------------------------------------------------
cap = ThreadedCamera(6) # for linux pc
print("CAMERA RUNNING")
symmetric_shapes = ["circle", "pentagon", "cube", "triangle", "star", "hexagon", "octagon", "heptagon", "square"]

latest_s = None 
is_symmetric = False
latest_class = None
prev_class = None
enable_control = False 
latest_message = {"status": "no data"} 

print("Started in MANUAL mode. Press 't' to toggle Autonomous Control.")

# --- PROFILER SETUP ---
frame_count = 0
profiler = {
    "1_camera": 0.0,
    "2_yolo": 0.0,
    "3_moments": 0.0,
    "4_control": 0.0,
    "5_zmq": 0.0,
    "6_gui": 0.0,
    "total": 0.0
}

while True:
    t_loop_start = time.perf_counter()

    # ---------------------------------------------------------
    # STAGE 1: Camera Read
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    ret, frame = cap.read()
    if not ret: break
    profiler["1_camera"] = (time.perf_counter() - t0)

    h, w, _ = frame.shape
    display_mask = np.zeros((h, w), dtype=np.uint8)

    # ---------------------------------------------------------
    # STAGE 2: YOLO Inference
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=0.3, verbose=False)
    profiler["2_yolo"] = (time.perf_counter() - t0)

    # ---------------------------------------------------------
    # STAGE 3: Post-processing & Moments
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    if len(results[0].boxes) > 0:
        class_id = int(results[0].boxes[0].cls[0].item())
        latest_class = model.names[class_id]
        is_symmetric = latest_class.lower() in symmetric_shapes

        if latest_class != prev_class: 
            reset_orientation_smoothing()
            prev_class = latest_class

        raw_mask = results[0].masks.data[0].cpu().numpy()
        mask_u8 = (raw_mask * 255).astype(np.uint8)
        display_mask = cv2.resize(mask_u8, (w, h))
        
        latest_s = get_img_moments(display_mask, is_symmetric, frame)

        cv2.drawContours(frame, [max(cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)], -1, (0, 255, 0), 2)
        cv2.putText(frame, f"Class: {latest_class}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    profiler["3_moments"] = (time.perf_counter() - t0)

    # ---------------------------------------------------------
    # STAGE 4: Control Law & Payload
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    if len(results[0].boxes) > 0:
        if latest_class not in gt_cache:
            gt_cache[latest_class] = load_ground_truth(latest_class)
        
        gt_features = gt_cache.get(latest_class)
        
        if gt_features is not None and latest_s is not None:
            visualize_error(frame, latest_s, gt_features, is_symmetric)
            v = control_law(latest_s, gt_features, is_symmetric)
            
            if enable_control:
                latest_message = {
                    "status": "success",
                    "velocities": {
                        "vx": float(v[0]), "vy": float(v[1]), "vz": float(v[2]),
                        "wx": float(v[3]), "wy": float(v[4]), "wz": float(v[5])
                    }
                }
            else:
                latest_message = {"status": "no data"}
        else:
            cv2.putText(frame, f"NO GROUND TRUTH FOR: {latest_class.upper()}", 
                        (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            latest_message = {"status": "no data"}
            
    else:
        reset_orientation_smoothing() 
        prev_class = None 
        latest_s = None
        latest_message = {"status": "no data"}
    profiler["4_control"] = (time.perf_counter() - t0)

    

    # ---------------------------------------------------------
    # STAGE 5: ZMQ Networking
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    try:
        incoming = socket.recv_json()
        if isinstance(incoming, dict) and "ping" in incoming.keys():
            socket.send_json({"pong": True})
        else:
            socket.send_json(latest_message)
    except zmq.Again:
        break 
    except Exception:
        pass  
    profiler["5_zmq"] = (time.perf_counter() - t0)

    # ---------------------------------------------------------
    # STAGE 6: GUI Rendering
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    mode_text = "MODE: AUTONOMOUS (SENDING)" if enable_control else "MODE: MANUAL (OBSERVING)"
    mode_color = (0, 255, 0) if enable_control else (0, 165, 255) 
    cv2.putText(frame, mode_text, (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    print("hi")
    time.sleep(0.5)
    cv2.imshow("1. Visual Servoing Target & Error", frame)
    cv2.imshow("2. Binary Segment", display_mask)

    del results

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and latest_s is not None:
        save_ground_truth(frame, latest_s, is_symmetric, latest_class)
    elif key == ord('t'):
        enable_control = not enable_control
        print(f"\n[MODE SWITCH] Autonomous Control is now: {'ON' if enable_control else 'OFF'}")
    profiler["6_gui"] = (time.perf_counter() - t0)

    # --- PROFILER REPORTING ---

    print(f"\n--- PROFILER REPORT ---")
    print(f"1. Camera Read: {profiler['1_camera']:.5f} sec")
    print(f"2. YOLO Infer:  {profiler['2_yolo']:.5f} sec")
    print(f"3. Moments:     {profiler['3_moments']:.5f} sec")
    print(f"4. Control Law: {profiler['4_control']:.5f} sec")
    print(f"5. ZMQ Network: {profiler['5_zmq']:.5f} sec")
    print(f"6. OpenCV GUI:  {profiler['6_gui']:.5f} sec")
        

cap.stop()
cv2.destroyAllWindows()
