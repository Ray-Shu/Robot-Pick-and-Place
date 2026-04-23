import cv2
from ultralytics import YOLO
import numpy as np
import math
import json
import os
import zmq
import threading
import time
from collections import deque 

model = YOLO('runs/segment/train/weights/best.pt')
CONFIDENCE_THRESHOLD = 0.65
OUTPUT = "ground_truth/"
os.makedirs(OUTPUT, exist_ok=True)

RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

LAMBDA_GAIN = 0.04
AREA_EPSILON = 1e-6
VZ_MIN_MAGNITUDE = 0.003
VZ_MAX_MAGNITUDE = 0.2

raw_writer = None
mask_writer = None

video_fps = 20.0
VZ_DEADBAND = 0.001

class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

        # removing exposure for camera frames stability
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.running = True

        self.ret = False
        self.frame = None

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()


    def update(self):
        while self.running:
            ret, frame = self.cap.read()

            with self.lock:
                self.ret = ret
                self.frame = frame if ret else None

            if ret:
                self.ready.set()

    def read(self, wait_for_first_frame=True, timeout=2.0):
        if wait_for_first_frame and not self.ready.wait(timeout):
            return False, None

        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
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

socket.bind("tcp://*:5555")  # publish on port 5555

# ----------------------------------------------------------
# Finding bucket/bowl/container to drop the toy object in 
# ----------------------------------------------------------

BOWL_CLASS_NAME = "circle"
SEARCH_HISTORY_LEN = 6
SEARCH_CLASS_CONFIRMATIONS = 5
MIN_BOWL_AREA_PX = 5000
MAX_CENTER_JUMP_PX = 35
recent_search_classes = deque(maxlen=SEARCH_HISTORY_LEN)
recent_search_centroids = deque(maxlen=SEARCH_HISTORY_LEN)

def reset_search_history():
    recent_search_classes.clear()
    recent_search_centroids.clear()


def update_bowl_search_history(latest_class, latest_s):
    """
    latest_s is expected as [cx, cy, area, alpha] or None
    """
    if latest_s is None:
        recent_search_classes.append(None)
        recent_search_centroids.append(None)
        return

    cx, cy, area, _ = latest_s

    if latest_class == BOWL_CLASS_NAME and area >= MIN_BOWL_AREA_PX:
        recent_search_classes.append(BOWL_CLASS_NAME)
        recent_search_centroids.append((float(cx), float(cy)))
    else:
        recent_search_classes.append(latest_class)
        recent_search_centroids.append(None)


def bowl_detection_is_stable():
    if len(recent_search_classes) < SEARCH_HISTORY_LEN:
        return False

    circle_count = sum(c == BOWL_CLASS_NAME for c in recent_search_classes)
    if circle_count < SEARCH_CLASS_CONFIRMATIONS:
        return False

    valid_centroids = [c for c in recent_search_centroids if c is not None]
    if len(valid_centroids) < SEARCH_CLASS_CONFIRMATIONS:
        return False

    xs = [c[0] for c in valid_centroids]
    ys = [c[1] for c in valid_centroids]

    if max(xs) - min(xs) > MAX_CENTER_JUMP_PX:
        return False

    if max(ys) - min(ys) > MAX_CENTER_JUMP_PX:
        return False

    return True


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

    L = compute_interaction_matrix(latest_s, is_symmetric) 
    L_pinv = np.linalg.pinv(L) 
    #print(L_pinv.shape, e.shape)
    v = -LAMBDA_GAIN * L_pinv @ e
    v[2] = shape_vertical_velocity(v[2])
    
    print(e)

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

def compute_interaction_matrix(s, is_symmetric, Z=1.0):
    # first run under the assumption that image plane is parallel to workshop plane 
    # this means that A = B = C = 0 

    cx, cy, area, alpha = s
    x,y = normalize_coords(cx, cy)

    L_xg = np.array([-1/Z, 0,    x/Z,             x*y,      -(1 + x**2), y])
    L_yg = np.array([0,   -1/Z,  y/Z,             1 + y**2, -x*y,       -x])
    L_area = np.array([0, 0, area*3/Z - area, 3*area*y, -3*area*x, 0])
    # We control with log(area / area*), so scale the area row accordingly.
    L_a = L_area / max(area, AREA_EPSILON)

    if not is_symmetric:
        row4 = np.array([0, 0, 0, 0, 0, -1])
        L = np.vstack([L_xg, L_yg, L_a, row4])
    else: 
        L = np.vstack([L_xg, L_yg, L_a])

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

current_state = None 

latest_s = None 
is_symmetric = False
latest_class = None
prev_class = None
enable_control = False

TRANS_THRESHOLD = 10
AREA_THRESHOLD = 1500   
#ROT_THRESHOLD = 0.05    
GRIPPER_OFFSET_M = 0.07 # 7cm
APPROACH_VEL = 0.01 # 1cm      

camera_read_failures = 0 
MAX_READ_FAILURES = 30

try: 
    while True:
        # retry reading cam frame a couple times before deciding fatal failure
        ret, frame = cap.read()

        if not ret:
            camera_read_failures += 1
            print(f"Camera read failed ({camera_read_failures}/{MAX_READ_FAILURES})")
            if camera_read_failures >= MAX_READ_FAILURES:
                print("Too many camera read failures, exiting.")
                break
            cv2.waitKey(1)
            continue
        camera_read_failures = 0

        # if theres no message incoming (ie im using controller, code loop will still continue)
        reply_to_request = False # Track if we need to reply to a request
        
        try:
            incoming = socket.recv_json(flags=zmq.NOBLOCK)
            state = incoming.get("vision_state")
            
            if state == "ping":
                socket.send_json({"pong": True}) 
                continue # Skip to next loop iteration
            
            if state == "get_velocities":
                # vision node is required to send velocity commands 
                reply_to_request = True 
                current_state = "tracking"
        
            if state == "search_bowl":
                bowl_found = bowl_detection_is_stable()

                socket.send_json({
                    "status": "success",
                    "found_bowl": bowl_found,
                })
                continue
        
        except zmq.Again:
            # No request from the controller? No problem. 
            # We just keep going to update the camera feed.
            pass
        
    
        h, w, _ = frame.shape   # (h, w) = 480 x 640 

        # record and save frame data for raw and masked 
        if raw_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            raw_path = os.path.join(RECORDINGS_DIR, "camera_view.mp4")
            mask_path = os.path.join(RECORDINGS_DIR, "binary_segment.mp4")

            raw_writer = cv2.VideoWriter(raw_path, fourcc, video_fps, (w, h))
            mask_writer = cv2.VideoWriter(mask_path, fourcc, video_fps, (w, h))

        display_mask = np.zeros((h, w), dtype=np.uint8)

        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=0.3, verbose=False)

        message = {"no data": True}
        if len(results[0].boxes) > 0 and results[0].masks is not None and len(results[0].masks.data) > 0:
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
            update_bowl_search_history(latest_class, latest_s)

        
            contours, _ = cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, f"Class: {latest_class}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ---------------------------------------------------------
            # Ground Truth Loading & Error Calculation (THE FAILSAFE)
            # ---------------------------------------------------------
            # 1. If we haven't checked for this shape's file yet, try to load it
            if latest_class not in gt_cache:
                gt_cache[latest_class] = load_ground_truth(latest_class)
            
            # 2. Safely retrieve the features (returns None if missing/corrupted)
            gt_features = gt_cache.get(latest_class)
            
            # 3. The Switch: Calculate error ONLY if we have both live moments AND ground truth
            if gt_features is not None and latest_s is not None:
                visualize_error(frame, latest_s, gt_features, is_symmetric)
            else:
                # Failsafe UI: Let the user know the loop is running, but no error is calculated
                cv2.putText(frame, f"NO GROUND TRUTH FOR: {latest_class.upper()}", 
                            (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # control law to get camera velocity
            if current_state == "tracking": 
                if latest_s is not None and gt_features is not None and enable_control:
                    v = control_law(latest_s, gt_features, is_symmetric)
                    """
                    Camera Frame Notes: 
                    vx: positive = right
                    vy: positive = backward  
                    vz: positive = down 
                    wx: positive = makes the camera look vertically up
                    wy: positive = makes the camera look right 
                    wz: positive = makes the camera tilt rightwards
                    """

                    """
                    Message Form: 
                    message = {
                        "status"        : "success" 
                        "type"          : "tracking" / "aligned" 
                        "velocities"    : v array 
                        "move_duration" : time in m/s
                    }
                    """

                    cx, cy, area, _ = latest_s
                    gt_x, gt_y = gt_features["centroid_x"], gt_features["centroid_y"]
                    gt_area = gt_features["area"]
                    
                    err_dist = math.sqrt((cx - gt_x)**2 + (cy - gt_y)**2)
                    err_area = abs(area - gt_area)

                    print("Err dist:", err_dist)
                    print("Err area:", err_area)
                    
                    is_aligned = err_dist < TRANS_THRESHOLD and err_area < AREA_THRESHOLD # Example px thresholds

                    if is_aligned:
                        contours, _ = cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        object_width_px = None
                        object_width_m = None
                        object_span_px = None
                        grasp_angle_deg = None

                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)

                            # Fit the smallest rotated rectangle around the segmented object
                            rect = cv2.minAreaRect(largest_contour)
                            (center_x, center_y), (side_a, side_b), angle_deg = rect

                            # The smaller side is the grasp width (width is in m)
                            object_width_m = float(min(side_a, side_b)) / FOCAL_LENGTH

                            # The larger side is the long span of the object
                            object_span_px = float(max(side_a, side_b))

                            grasp_angle_deg = float(angle_deg)

                            print(f"Aligned object width: {object_width_m:.2f} m")
                            print(f"Aligned object span:  {object_span_px:.2f} px")
                            print(f"Grasp rect angle:     {grasp_angle_deg:.2f} deg")

                        message = {
                            "status": "success",
                            "type": "aligned",
                            "velocities": {
                                "vx": float(0),
                                "vy": float(-APPROACH_VEL),
                                "vz": float(0),
                                "wx": float(0),
                                "wy": float(0),
                                "wz": float(0)
                            },

                            "move_duration": GRIPPER_OFFSET_M / APPROACH_VEL,
                            "object_width_m": object_width_m,
                            "object_span_px": object_span_px,
                            "grasp_rect_angle_deg": grasp_angle_deg
                        }

                        print("vz", message["velocities"])
                        print("move dur", message["move_duration"])
                        print("width (m)", message["object_width_m"])
                        print("object span (px)", message["object_span_px"])
                        print("rect angle deg idk", message["grasp_rect_angle_deg"])
                        print(reply_to_request)

                    else: 
                        message = {
                            "status": "success",
                            "type"  : "tracking",
                            "velocities": {
                                "vx": float(v[0]),
                                "vy": float(v[1]),
                                "vz": float(v[2]),
                                "wx": float(v[3]),
                                "wy": float(v[4]),
                                "wz": float(v[5])
                            }
                        }
                else:
                    reset_orientation_smoothing() 
                    prev_class = None 
                    latest_s = None
                    # If no target is seen or no ground truth exists
                    message = {
                        "status": "no data"
                    }

            elif current_state == "perimeter_sweep":
                message = {
                    "status": "success",
                    "type": "perimeter_sweep",
                    "send_joint_angles": [ 

                    ]
                }
        else: 
            latest_s = None
            latest_class = None
            reset_orientation_smoothing()
            prev_class = None
            update_bowl_search_history(None, None)

        if reply_to_request: 
            socket.send_json(message)    

        # shows manual or auto mode (press "t" to switch)
        mode_text = "MODE: AUTONOMOUS" if enable_control else "MODE: MANUAL"
        mode_color = (0, 255, 0) if enable_control else (0, 165, 255) 
        cv2.putText(frame, mode_text, (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)

        display_mask_bgr = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)
        raw_writer.write(frame)
        mask_writer.write(display_mask_bgr)

        cv2.imshow("1. Visual Servoing Target & Error", frame)
        cv2.imshow("2. Binary Segment", display_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and latest_s is not None:
            save_ground_truth(frame, latest_s, is_symmetric, latest_class)
        elif key == ord('t'):
            enable_control = not enable_control

except KeyboardInterrupt:
    print("\nKeyboard interrupt received, closing cleanly...")

finally:
    if raw_writer is not None:
        raw_writer.release()
    if mask_writer is not None:
        mask_writer.release()
    cap.stop()
    cv2.destroyAllWindows()
