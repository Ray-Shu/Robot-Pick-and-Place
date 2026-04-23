"""
Synthetic Geometric Shape Dataset Generator  —  YOLO Segmentation Format
=========================================================================
Generates images + YOLO-seg polygon labels for:
  0: ellipse   (ellipsoid / cylinder-capped)
  1: cross     (extruded plus prism)
  2: pentagon  (pentagonal prism)
  3: cube      (square prism)
  4: triangle  (triangular prism)
  5: star      (extruded star prism)
  6: hexagon   (hexagonal prism)
  7: circle    (cylinder)

Label format  (YOLO segmentation):
  <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>   (all values normalised 0-1)

The silhouette polygon is extracted via cv2.findContours on a per-shape
binary mask, so it is pixel-accurate for every shape regardless of complexity.
This mask is also the exact input you need for Chaumette moment-based
visual servoing:

    results  = model(frame)                             # YOLO-seg inference
    mask     = results[0].masks.data[i].cpu().numpy()   # float32 H x W
    mask_u8  = (mask * 255).astype(np.uint8)
    M        = cv2.moments(mask_u8)                     # image moments
    cx       = M['m10'] / M['m00']                      # centroid x
    cy       = M['m01'] / M['m00']                      # centroid y
    # higher-order moments -> orientation/shape features for Chaumette
    # interaction matrix L_s

Output structure:
  dataset/
    images/train/
    images/val/
    labels/train/    <- YOLO seg polygons (.txt)
    labels/val/
    data.yaml        <- ready for: yolo segment train ...
"""

import cv2
import numpy as np
import os
import math
import random
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR       = "dataset"
IMG_SIZE         = 640
NUM_IMAGES       = 1000
VAL_SPLIT        = 0.2
SHAPES_PER_IMAGE = (1, 4)

CLASSES = ["ellipse", "cross", "pentagon", "cube", "triangle", "star", "hexagon", "circle"]

# ── General Helpers ────────────────────────────────────────────────────────────

def random_color():
    return tuple(int(x) for x in np.random.randint(30, 255, 3))

def shade(color, delta):
    return tuple(int(np.clip(c + delta, 0, 255)) for c in color)

def random_background(size):
    mode = random.choice(["noise", "gradient", "solid"])
    img  = np.zeros((size, size, 3), dtype=np.uint8)
    if mode == "noise":
        img = np.random.randint(20, 180, (size, size, 3), dtype=np.uint8)
    elif mode == "gradient":
        c1 = np.array(random_color(), dtype=np.float32)
        c2 = np.array(random_color(), dtype=np.float32)
        for i in range(size):
            img[i] = (c1 + (c2 - c1) * i / size).astype(np.uint8)
    else:
        img[:] = random_color()
    return img

def random_transform_pts(pts, cx, cy, angle=None):
    """Rotate points around (cx, cy) by a random (or given) angle."""
    if angle is None:
        angle = random.uniform(0, 360)
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        out.append([cx + dx * cos_a - dy * sin_a,
                    cy + dx * sin_a + dy * cos_a])
    return np.array(out, dtype=np.int32)

def poly_points(cx, cy, n_sides, radius, start_angle=0):
    pts = []
    for i in range(n_sides):
        a = math.radians(start_angle + 360 * i / n_sides)
        pts.append([int(cx + radius * math.cos(a)),
                    int(cy + radius * math.sin(a))])
    return np.array(pts, dtype=np.int32)

def iso_offset(scale):
    """Extrusion vector in a fully random direction."""
    depth = int(scale * random.uniform(0.25, 0.45))
    angle = random.uniform(0, 2 * math.pi)
    return np.array([int(depth * math.cos(angle)),
                     int(depth * math.sin(angle))], dtype=np.int32)

# ── Segmentation Helpers ───────────────────────────────────────────────────────

def contour_from_mask(mask):
    """
    Return the largest external contour from a binary mask as an (N,2) int32
    array, or None if nothing meaningful is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    return cnt.reshape(-1, 2)

def seg_label(class_id, contour, img_size):
    """
    Convert a contour (N,2) to a YOLO-seg annotation string.
    Format: class_id x1 y1 x2 y2 ... xn yn  (coords normalised to [0,1])
    """
    pts  = contour.astype(float)
    xs   = np.clip(pts[:, 0] / img_size, 0.0, 1.0)
    ys   = np.clip(pts[:, 1] / img_size, 0.0, 1.0)
    flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
    return f"{class_id} {flat}"

def bbox_valid(contour, img_size, min_frac=0.02):
    """Return True if the contour bounding box is large enough to be useful."""
    x, y, w, h = cv2.boundingRect(contour)
    return (w / img_size) > min_frac and (h / img_size) > min_frac

# ── 3-D Drawing Helper ─────────────────────────────────────────────────────────
'''
def draw_prism(img, mask, front_pts, offset, base_color):
    """
    Extrude a polygon into a 3-D prism on both img (colour) and mask (binary).
    The mask is filled with the convex hull of all front+back points so that
    cv2.moments() receives a solid, contiguous silhouette region.
    """
    n         = len(front_pts)
    back_pts  = (front_pts + offset).astype(np.int32)
    col_front = base_color
    col_side  = shade(base_color, -55)
    col_back  = shade(base_color, -90)
    col_edge  = shade(base_color, -120)
    thickness = random.randint(1, 3)

    # colour image: back -> sides -> front
    cv2.fillPoly(img, [back_pts], col_back)
    for i in range(n):
        j    = (i + 1) % n
        quad = np.array([front_pts[i], front_pts[j],
                         back_pts[j],  back_pts[i]], dtype=np.int32)
        cv2.fillPoly(img,  [quad], col_side)
        cv2.polylines(img, [quad], True, col_edge, thickness)
    cv2.fillPoly(img,  [front_pts], col_front)
    cv2.polylines(img, [front_pts], True, col_edge, thickness)
    cv2.polylines(img, [back_pts],  True, col_edge, thickness)

    # binary mask: convex hull of entire visible silhouette
    all_pts = np.vstack([front_pts, back_pts])
    hull    = cv2.convexHull(all_pts)
    cv2.fillPoly(mask, [hull], 255)
'''

def draw_prism(img, mask, front_pts, offset, base_color):
    """
    Extrude a polygon into a 3-D prism on both img (colour) and mask (binary).
    The mask exactly mirrors the drawn geometry for tight segmentation.
    """
    n         = len(front_pts)
    back_pts  = (front_pts + offset).astype(np.int32)
    col_front = base_color
    col_side  = shade(base_color, -55)
    col_back  = shade(base_color, -90)
    col_edge  = shade(base_color, -120)
    thickness = random.randint(1, 3)

    # 1. Back face
    cv2.fillPoly(img, [back_pts], col_back)
    cv2.fillPoly(mask, [back_pts], 255)
    
    # 2. Side faces
    for i in range(n):
        j    = (i + 1) % n
        quad = np.array([front_pts[i], front_pts[j],
                         back_pts[j],  back_pts[i]], dtype=np.int32)
        
        cv2.fillPoly(img,  [quad], col_side)
        cv2.fillPoly(mask, [quad], 255)
        
        cv2.polylines(img, [quad], True, col_edge, thickness)
        cv2.polylines(mask, [quad], True, 255, thickness)
        
    # 3. Front face
    cv2.fillPoly(img,  [front_pts], col_front)
    cv2.fillPoly(mask, [front_pts], 255)
    
    cv2.polylines(img, [front_pts], True, col_edge, thickness)
    cv2.polylines(mask, [front_pts], True, 255, thickness)
    
    cv2.polylines(img, [back_pts],  True, col_edge, thickness)
    cv2.polylines(mask, [back_pts], True, 255, thickness)

# ── Shape Drawers ──────────────────────────────────────────────────────────────
# Each drawer renders onto both `img` (colour) and `mask` (uint8 binary).
# The caller extracts the segmentation contour from `mask` afterwards.

def draw_ellipse(img, mask, cx, cy, scale):
    rx    = int(random.uniform(0.5, 1.0) * scale)
    ry    = int(random.uniform(0.35, 0.7) * scale)
    tilt  = random.randint(0, 179)
    off   = iso_offset(scale)
    color = random_color()
    col_side = shade(color, -60)
    col_back = shade(color, -100)
    col_edge = shade(color, -130)
    thickness = random.randint(1, 3)

    bcx, bcy = cx + off[0], cy + off[1]
    rad_t    = math.radians(tilt)
    ex = int(rx * math.cos(rad_t))
    ey = int(rx * math.sin(rad_t))

    side_poly = np.array([[cx  - ex, cy  - ey],
                           [bcx - ex, bcy - ey],
                           [bcx + ex, bcy + ey],
                           [cx  + ex, cy  + ey]], dtype=np.int32)

    # colour image
    cv2.fillPoly(img,  [side_poly], col_side)
    cv2.ellipse(img, (bcx, bcy), (rx, ry), tilt, 0, 360, col_back, -1)
    cv2.ellipse(img, (bcx, bcy), (rx, ry), tilt, 0, 360, col_edge, thickness)
    cv2.ellipse(img, (cx,  cy),  (rx, ry), tilt, 0, 360, color,    -1)
    cv2.ellipse(img, (cx,  cy),  (rx, ry), tilt, 0, 360, col_edge, thickness)
    cv2.line(img, (cx - ex, cy - ey), (bcx - ex, bcy - ey), col_edge, thickness)
    cv2.line(img, (cx + ex, cy + ey), (bcx + ex, bcy + ey), col_edge, thickness)

    # mask: side band + both elliptical caps
    cv2.fillPoly(mask,  [side_poly], 255)
    cv2.ellipse(mask, (bcx, bcy), (rx, ry), tilt, 0, 360, 255, -1)
    cv2.ellipse(mask, (cx,  cy),  (rx, ry), tilt, 0, 360, 255, -1)

def draw_cross(img, mask, cx, cy, scale):
    arm   = int(scale * random.uniform(0.6, 1.0))
    thick = int(scale * random.uniform(0.22, 0.38))
    h     = thick // 2
    front = np.array([
        [cx - h,   cy - arm], [cx + h,   cy - arm],
        [cx + h,   cy - h  ], [cx + arm, cy - h  ],
        [cx + arm, cy + h  ], [cx + h,   cy + h  ],
        [cx + h,   cy + arm], [cx - h,   cy + arm],
        [cx - h,   cy + h  ], [cx - arm, cy + h  ],
        [cx - arm, cy - h  ], [cx - h,   cy - h  ],
    ], dtype=np.int32)
    front = random_transform_pts(front, cx, cy)
    draw_prism(img, mask, front, iso_offset(scale), random_color())

def draw_pentagon(img, mask, cx, cy, scale):
    radius = int(scale * random.uniform(0.5, 1.0))
    front  = poly_points(cx, cy, 5, radius, start_angle=-90)
    front  = random_transform_pts(front, cx, cy)
    draw_prism(img, mask, front, iso_offset(scale), random_color())

def draw_cube(img, mask, cx, cy, scale):
    s     = int(scale * random.uniform(0.5, 0.9))
    front = poly_points(cx, cy, 4, s, start_angle=0)
    front = random_transform_pts(front, cx, cy)
    draw_prism(img, mask, front, iso_offset(scale), random_color())

def draw_triangle(img, mask, cx, cy, scale):
    radius = int(scale * random.uniform(0.5, 1.0))
    front  = poly_points(cx, cy, 3, radius, start_angle=-90)
    front  = random_transform_pts(front, cx, cy)
    draw_prism(img, mask, front, iso_offset(scale), random_color())

def draw_star(img, mask, cx, cy, scale):
    outer = int(scale * random.uniform(0.6, 1.0))
    inner = int(outer * random.uniform(0.35, 0.55))
    pts   = []
    for i in range(10):
        r = outer if i % 2 == 0 else inner
        a = math.radians(-90 + 36 * i)
        pts.append([int(cx + r * math.cos(a)), int(cy + r * math.sin(a))])
    front = random_transform_pts(np.array(pts, dtype=np.int32), cx, cy)
    draw_prism(img, mask, front, iso_offset(scale), random_color())

def draw_hexagon(img, mask, cx, cy, scale):
    radius = int(scale * random.uniform(0.5, 1.0))
    front  = poly_points(cx, cy, 6, radius, start_angle=0)
    front  = random_transform_pts(front, cx, cy)
    draw_prism(img, mask, front, iso_offset(scale), random_color())

def draw_circle(img, mask, cx, cy, scale):
    radius    = int(scale * random.uniform(0.4, 1.0))
    tilt      = random.randint(0, 179)
    off       = iso_offset(scale)
    color     = random_color()
    col_side  = shade(color, -60)
    col_back  = shade(color, -100)
    col_edge  = shade(color, -130)
    thickness = random.randint(1, 3)

    bcx, bcy = cx + off[0], cy + off[1]
    rad_t    = math.radians(tilt)
    ex = int(radius * math.cos(rad_t))
    ey = int(radius * math.sin(rad_t))

    side_poly = np.array([[cx  - ex, cy  - ey],
                           [bcx - ex, bcy - ey],
                           [bcx + ex, bcy + ey],
                           [cx  + ex, cy  + ey]], dtype=np.int32)

    # colour image
    cv2.fillPoly(img,  [side_poly], col_side)
    cv2.circle(img, (bcx, bcy), radius, col_back, -1)
    cv2.circle(img, (bcx, bcy), radius, col_edge, thickness)
    cv2.circle(img, (cx,  cy),  radius, color,    -1)
    cv2.circle(img, (cx,  cy),  radius, col_edge, thickness)
    cv2.line(img, (cx - ex, cy - ey), (bcx - ex, bcy - ey), col_edge, thickness)
    cv2.line(img, (cx + ex, cy + ey), (bcx + ex, bcy + ey), col_edge, thickness)

    # mask
    cv2.fillPoly(mask,  [side_poly], 255)
    cv2.circle(mask, (bcx, bcy), radius, 255, -1)
    cv2.circle(mask, (cx,  cy),  radius, 255, -1)


DRAWERS = [
    draw_ellipse, draw_cross,   draw_pentagon, draw_cube,
    draw_triangle, draw_star,   draw_hexagon,  draw_circle,
]

# ── Image Generator ────────────────────────────────────────────────────────────

def generate_image():
    img         = random_background(IMG_SIZE)
    annotations = []

    n_shapes = random.randint(*SHAPES_PER_IMAGE)
    for _ in range(n_shapes):
        class_id = random.randint(0, len(CLASSES) - 1)
        scale    = random.randint(40, 120)
        margin   = scale + 15
        cx = random.randint(margin, IMG_SIZE - margin)
        cy = random.randint(margin, IMG_SIZE - margin)

        # Fresh per-shape mask — keeps silhouettes independent
        shape_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        DRAWERS[class_id](img, shape_mask, cx, cy, scale)

        contour = contour_from_mask(shape_mask)
        if contour is not None and bbox_valid(contour, IMG_SIZE):
            annotations.append(seg_label(class_id, contour, IMG_SIZE))

    return img, annotations

# ── Dataset Builder ────────────────────────────────────────────────────────────

def build_dataset():
    for split in ("train", "val"):
        Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)

    n_val   = int(NUM_IMAGES * VAL_SPLIT)
    n_train = NUM_IMAGES - n_val

    for i in range(NUM_IMAGES):
        split            = "val" if i < n_val else "train"
        img, annotations = generate_image()
        name             = f"{i:05d}"

        cv2.imwrite(f"{OUTPUT_DIR}/images/{split}/{name}.jpg", img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(f"{OUTPUT_DIR}/labels/{split}/{name}.txt", "w") as f:
            f.write("\n".join(annotations))

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{NUM_IMAGES} images...")

    yaml_content = f"""\
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val:   images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n Done!  Dataset saved to '{OUTPUT_DIR}/'")
    print(f"   Train : {n_train} images  |  Val : {n_val} images")
    print(f"   Classes: {CLASSES}")
    print(f"\n   Train segmentation model:")
    print(f"   yolo segment train data={OUTPUT_DIR}/data.yaml model=yolov8n-seg.pt epochs=50 imgsz=640")

# ── Preview ────────────────────────────────────────────────────────────────────

def preview(n=4):
    """Save a preview grid with green contour overlays so you can verify masks."""
    cell = IMG_SIZE // 2
    grid = np.zeros((cell * n, cell * n, 3), dtype=np.uint8)
    for row in range(n):
        for col in range(n):
            img    = random_background(IMG_SIZE)
            scale  = random.randint(60, 140)
            margin = scale + 15
            cx  = random.randint(margin, IMG_SIZE - margin)
            cy  = random.randint(margin, IMG_SIZE - margin)
            cid = random.randint(0, len(CLASSES) - 1)

            shape_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            DRAWERS[cid](img, shape_mask, cx, cy, scale)

            cnt = contour_from_mask(shape_mask)
            if cnt is not None:
                cv2.polylines(img, [cnt], True, (0, 255, 0), 2)

            cv2.putText(img, CLASSES[cid], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            grid[row*cell:(row+1)*cell, col*cell:(col+1)*cell] = cv2.resize(img, (cell, cell))

    cv2.imwrite("preview.jpg", grid)
    print("Preview saved to preview.jpg  (green = segmentation contour)")

# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic shape dataset generator (YOLO-seg)")
    parser.add_argument("--preview", action="store_true",
                        help="Save a preview grid with mask overlays instead of full dataset")
    parser.add_argument("--n", type=int, default=NUM_IMAGES,
                        help=f"Number of images to generate (default {NUM_IMAGES})")
    args = parser.parse_args()

    if args.preview:
        preview()
    else:
        NUM_IMAGES = args.n
        build_dataset()