import cv2
from ultralytics import YOLO
import numpy as np
import math
import json

model = YOLO('runs/segment/train/weights/best.pt')
CONFIDENCE_THRESHOLD = 0.70
OUTPUT = "ground_truth/"

CLASSES = ["CIRCLE", 
           "CROSS",
           "HEPTAGON", 
           "HEXAGON",
           "OCTAGON",
           "PENTAGON", 
           "QUARTER_CIRCLE",
           "RECTANGLE", 
           "SEMICIRCLE", 
           "SQUARE",
           "STAR",
           "TRAPEZOID",
           "TRIANGLE"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def save_ground_truth(frame, s, is_symmetric): 
    '''
    Saves the ground truth values of image moments.
    Args:
        frame: the current frame to be saved
        s: the image moment features consisting: [centroid_x, centroid_y, area, alpha], 
        where alpha is the z-rotation of the camera in degrees

        is_symmetric: whether the object being detected is a symmetric object (to know when to use alpha)
    '''
    if not is_symmetric:
        [cx, cy, area, alpha_rad] = s
        alpha_deg = math.degrees(alpha_rad)

        features = {
            "area": area,
            "centroid_x": cx,
            "centroid_y": cy,
            "alpha_degrees": alpha_deg,
        }
    else:
        cx, cy, area, _ = s 
        features = {
            "area": area,
            "centroid_x": cx,
            "centroid_y": cy,
        }

    cv2.imwrite(OUTPUT+"annotated_ground_truth.jpg", frame)
    print("Saved visual verification to 'annotated_ground_truth.jpg'")

    with open(OUTPUT+"pentagon_features.json", "w") as f:
        json.dump(features, f, indent=4)
    print("Saved numerical features to 'target_features.json'")

def visualize_img_moments(frame, s, is_symmetric):
    '''
    Outputs image moments numbers on the screen.
    '''
    [centroid_x, centroid_y, area, alpha_in_rads] = s
    
    # Draw the Centroid (Red dot)
    cv2.circle(frame, (int(centroid_x), int(centroid_y)), 7, (0, 0, 255), 1)

    # Display the values on the screen
    cv2.putText(frame, f"Area: {int(area)} px", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Centroid: {centroid_x}, {centroid_y}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    
    if not is_symmetric: 
        # Draw the Orientation Line (Blue line indicating the angle)
        line_length = 75
        end_x = int(centroid_x + line_length * math.cos(alpha_in_rads))
        end_y = int(centroid_y + line_length * math.sin(alpha_in_rads))
        cv2.line(frame, (int(centroid_x), int(centroid_y)), (end_x, end_y), (255, 0, 0), 3)
        cv2.putText(frame, f"Angle: {math.degrees(alpha_in_rads):.1f} deg", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def get_img_moments(binary_mask, is_symmetric):
    '''
    Gets the image moments from Chaumette's paper where s = (centroid_x, centroid_y, area, camera_roll)
    '''
   

    M = cv2.moments(binary_mask)
    if M["m00"] != 0: 
        # Area 
        area = M["m00"]

        # Centroids 
        centroid_x = M["m10"] / M["m00"]
        centroid_y = M["m01"] / M["m00"]
        
        if is_symmetric:
            alpha_in_rads = None
            is_symmetric = True 
        else:
            # Orientation 
            alpha_smooth = 0.15  # Smoothing factor (0.0 to 1.0. Lower = smoother but slower to react)
            smooth_mu20 = None
            smooth_mu02 = None
            smooth_mu11 = None

            # Inside your loop, where you extract moments:
            mu20 = M["mu20"]
            mu02 = M["mu02"]
            mu11 = M["mu11"]

            # 1. Initialize or Smooth the Moments
            if smooth_mu20 is None:
                smooth_mu20, smooth_mu02, smooth_mu11 = mu20, mu02, mu11
            else:
                smooth_mu20 = alpha_smooth * mu20 + (1 - alpha_smooth) * smooth_mu20
                smooth_mu02 = alpha_smooth * mu02 + (1 - alpha_smooth) * smooth_mu02
                smooth_mu11 = alpha_smooth * mu11 + (1 - alpha_smooth) * smooth_mu11

            # 2. Calculate the Angle using the SMOOTHED moments
            alpha_in_rads = 0.5 * math.atan2(2 * smooth_mu11, smooth_mu20 - smooth_mu02) # z-rotation (alpha) in radians

        s = [centroid_x, centroid_y, area, alpha_in_rads]
        visualize_img_moments(frame, s, is_symmetric) 

        return s

is_symmetric = False
symmetric_shapes = ["circle", "pentagon", "cube", "triangle", "star", "hexagon"]

while True:
    ret, frame = cap.read()
    if not ret: break

    # Create a blank black mask for when nothing is detected
    h, w, _ = frame.shape
    display_mask = np.zeros((h, w), dtype=np.uint8)

    # 2. Inference
    results = model(frame, conf=0.80, iou=0.3, verbose=False)

    if len(results[0].boxes) > 0:
        best_box = results[0].boxes[0]
        class_id = int(best_box.cls[0].item())
        class_name = model.names[class_id]
        
        if class_name in symmetric_shapes:
            is_symmetric=True

        # Get the highest confidence mask
        # .data[0] is the mask tensor; .cpu().numpy() moves it to system RAM
        raw_mask = results[0].masks.data[0].cpu().numpy()
        
        # 3. Process Mask for Display
        # Scale 0.0-1.0 float to 0-255 uint8
        mask_u8 = (raw_mask * 255).astype(np.uint8)
        
        # Resize to match camera frame dimensions
        display_mask = cv2.resize(mask_u8, (w, h))
        s = get_img_moments(display_mask, is_symmetric)
        [centroid_x, centroid_y, area, alpha_in_rads] = s

        # 4. Draw outline on the COLOR frame for reference
        contours, _ = cv2.findContours(display_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # 5. Show both windows
    cv2.imshow("1. Original Feed (with Contour)", frame)
    cv2.imshow("2. Binary Segmentation Mask", display_mask)

    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == ord('s'):
        save_ground_truth(frame, s, is_symmetric)
       

cap.release()
cv2.destroyAllWindows()