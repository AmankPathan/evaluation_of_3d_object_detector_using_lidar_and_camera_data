import cv2
import numpy as np
from ultralytics import YOLO
import os
from glob import glob
import open3d as o3d
import json
import random
import pandas as pd
import matplotlib.pyplot as plt

def load_velodyne_points(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]

def parse_perspective_txt(perspective_path):
    K_00 = None
    R_rect_00 = None
    P_rect_00 = None
    with open(perspective_path, 'r') as f:
        for line in f:
            if line.startswith('K_00:'):
                K_00 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 3)
            if line.startswith('R_rect_00:'):
                R_rect_00 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 3)
            if line.startswith('P_rect_00:'):
                P_rect_00 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
    assert K_00 is not None and R_rect_00 is not None and P_rect_00 is not None, "Calibration parsing failed!"
    return K_00, R_rect_00, P_rect_00

def parse_cam_to_velo(calib_path):
    with open(calib_path, 'r') as f:
        line = f.readline()
        values = [float(x) for x in line.strip().split()]
        Tr = np.array(values).reshape(3, 4)
    return Tr

def project_lidar_to_image(pts_lidar, Tr, R_rect, P_rect):
    pts_hom = np.hstack((pts_lidar, np.ones((pts_lidar.shape[0], 1))))
    pts_cam = (Tr @ pts_hom.T).T
    pts_cam_rect = (R_rect @ pts_cam.T).T
    mask = pts_cam_rect[:, 2] > 0
    pts_cam_rect = pts_cam_rect[mask]
    pts_cam_rect_hom = np.hstack((pts_cam_rect, np.ones((pts_cam_rect.shape[0], 1))))
    pts_img = (P_rect @ pts_cam_rect_hom.T).T
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]
    return pts_img, mask

def get_bbox_json_path(base_name, bbox3d_folder):
    idx = str(int(base_name))
    bbox_json_name = f"BBoxes_{idx}.json"
    bbox_json_path = os.path.join(bbox3d_folder, bbox_json_name)
    return bbox_json_path

image_folder = r"C:\Users\Aman\Desktop\2nd sem\LRS\compulsary task\KITTI-360_sample\KITTI-360_sample\data_2d_raw\2013_05_28_drive_0000_sync\image_00\data_rect"
lidar_folder = r"C:\Users\Aman\Desktop\2nd sem\LRS\compulsary task\KITTI-360_sample\KITTI-360_sample\data_3d_raw\2013_05_28_drive_0000_sync\velodyne_points\data"
perspective_path = r"C:\Users\Aman\Desktop\2nd sem\LRS\compulsary task\KITTI-360_sample\KITTI-360_sample\calibration\perspective.txt"
calib_cam_to_velo_path = r"C:\Users\Aman\Desktop\2nd sem\LRS\compulsary task\KITTI-360_sample\KITTI-360_sample\calibration\calib_cam_to_velo.txt"
bbox3d_folder = r"C:\Users\Aman\Desktop\2nd sem\LRS\compulsary task\KITTI-360_sample\KITTI-360_sample\bboxes_3D_cam0"

detections_folder = "detections"
os.makedirs(detections_folder, exist_ok=True)
for f in glob(os.path.join(detections_folder, "*.png")):
    os.remove(f)

K_00, R_rect_00, P_rect_00 = parse_perspective_txt(perspective_path)
Tr_cam_to_velo = parse_cam_to_velo(calib_cam_to_velo_path)
Tr_cam_to_velo_4x4 = np.eye(4)
Tr_cam_to_velo_4x4[:3, :4] = Tr_cam_to_velo
Tr_velo_to_cam_4x4 = np.linalg.inv(Tr_cam_to_velo_4x4)
Tr_velo_to_cam = Tr_velo_to_cam_4x4[:3, :4]

model = YOLO("yolov8x-seg.pt")
car_class_id = 2  # COCO: 2 = car

image_paths = sorted(glob(os.path.join(image_folder, "*.png")))
target_size = (1408, 376)  # width, height from S_rect_00

all_eval_rows = []

for image_path in image_paths:  
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    assert image is not None, f"Image not found at {image_path}"
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    lidar_path = os.path.join(lidar_folder, base_name + ".bin")
    if not os.path.exists(lidar_path):
        print(f"Lidar file not found for {base_name}, skipping.")
        continue
    pts_lidar = load_velodyne_points(lidar_path)

    results = model(image, imgsz=1280)
    car_mask_total = np.zeros(image.shape[:2], dtype=bool)
    overlay = image.copy()
    car_boxes = []
    for r in results:
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            for mask, cls, conf, box in zip(masks, classes, confs, boxes):
                if cls == car_class_id:
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    kernel = np.ones((5, 5), np.uint8)
                    mask_dilated = cv2.dilate(mask_resized.astype(np.uint8), kernel, iterations=1)
                    car_mask_total |= mask_dilated.astype(bool)
                    x1, y1, x2, y2 = box
                    car_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(
                        overlay,
                        f"car {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )

    pts_img, mask_in_front = project_lidar_to_image(pts_lidar, Tr_velo_to_cam, R_rect_00, P_rect_00)
    pts_img = pts_img.astype(int)
    h, w = image.shape[:2]
    mask_in_front_indices = np.where(mask_in_front)[0]  # Indices in original pts_lidar that are in mask

    # Group lidar point indices per detected car bounding box (indices refer to original pts_lidar)
    lidar_indices_per_car_box = []
    for (x1, y1, x2, y2) in car_boxes:
        indices = []
        for idx, (u, v) in enumerate(pts_img):
            if 0 <= u < w and 0 <= v < h:
                if x1 <= u <= x2 and y1 <= v <= y2:
                    indices.append(mask_in_front_indices[idx])  # Store original index
        lidar_indices_per_car_box.append(indices)

    overlay[car_mask_total] = [255, 255, 255]  # White for car mask

    for idx, (u, v) in enumerate(pts_img):
        if 0 <= u < w and 0 <= v < h:
            in_box = any((x1 <= u <= x2 and y1 <= v <= y2) for (x1, y1, x2, y2) in car_boxes)
            if in_box:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
            cv2.circle(overlay, (u, v), 1, color, -1)

    alpha = 0.5
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    cv2.imwrite(f"{detections_folder}/{base_name}_fused.png", output)
    print(f"{base_name}: Fused image saved.")

    # Visualize the full point cloud for the image
    pts_hom = np.hstack((pts_lidar, np.ones((pts_lidar.shape[0], 1))))
    pts_cam = (Tr_velo_to_cam @ pts_hom.T).T
    projected_lidar_points = pts_cam[mask_in_front]
    colors = np.ones((projected_lidar_points.shape[0], 3)) * [0.4, 0.4, 0.4]

    neon_colors = [
        [0.0, 1.0, 0.0],   # Neon Green
        [1.0, 0.0, 1.0],   # Magenta
        [0.0, 1.0, 1.0],   # Cyan
        [1.0, 1.0, 0.0],   # Yellow
        [1.0, 0.5, 0.0],   # Orange
        [0.0, 0.5, 1.0],   # Blue-cyan
        [1.0, 0.0, 0.5],   # Pink
        [0.5, 1.0, 0.0],   # Lime
        [0.0, 1.0, 0.5],   # Aqua
        [1.0, 0.25, 0.7],  # Hot Pink
    ]
    car_colors = []
    for i, _ in enumerate(lidar_indices_per_car_box):
        color = neon_colors[i % len(neon_colors)]
        car_colors.append(color)

    # --- CORRECT: assign colors for each car's points ---
    # Map original indices to masked indices for coloring
    orig_idx_to_masked_idx = {orig_idx: i for i, orig_idx in enumerate(mask_in_front_indices)}
    for car_idx, indices in enumerate(lidar_indices_per_car_box):
        if len(indices) > 0:
            masked_indices = [orig_idx_to_masked_idx[idx] for idx in indices if idx in orig_idx_to_masked_idx]
            colors[masked_indices] = car_colors[car_idx]

    # --------- 3D BOUNDING BOXES FROM JSON ---------
    bbox_json_path = get_bbox_json_path(base_name, bbox3d_folder)
    line_sets = []
    det_idx_to_bbox_idx = {}
    if os.path.exists(bbox_json_path):
        with open(bbox_json_path, "r") as f:
            bbox_data = json.load(f)

        bbox_centers_3d = []
        bbox_centers_2d = []
        for car in bbox_data:
            if "corners_cam0" in car and car["corners_cam0"]:
                corners = np.array(car["corners_cam0"], dtype=np.float32)
                center_3d = np.mean(corners, axis=0)
                bbox_centers_3d.append(center_3d)
                center_3d_hom = np.hstack([center_3d, 1.0])
                center_cam = center_3d_hom
                center_img = (P_rect_00 @ center_cam[:4].reshape(4,1)).flatten()
                center_img = center_img[:2] / center_img[2]
                bbox_centers_2d.append(center_img)

        used_bbox = set()
        for det_idx, (x1, y1, x2, y2) in enumerate(car_boxes):
            box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            min_dist = float('inf')
            min_bbox_idx = -1
            for bbox_idx, bbox_center in enumerate(bbox_centers_2d):
                if bbox_idx in used_bbox:
                    continue
                dist = np.linalg.norm(box_center - bbox_center)
                if dist < min_dist:
                    min_dist = dist
                    min_bbox_idx = bbox_idx
            if min_bbox_idx >= 0 and len(lidar_indices_per_car_box[det_idx]) > 0:
                used_bbox.add(min_bbox_idx)
                det_idx_to_bbox_idx[det_idx] = min_bbox_idx
                car = bbox_data[min_bbox_idx]
                corners = np.array(car["corners_cam0"], dtype=np.float32)
                lines = [
                    [0, 1], [1, 3], [3, 2], [2, 0],  # bottom face
                    [4, 5], [5, 7], [7, 6], [6, 4],  # top face
                    [0, 5], [1, 4], [2, 7], [3, 6]   # verticals
                ]
                colors_box = [car_colors[det_idx] for _ in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors_box)
                line_sets.append(line_set)

    # --------- OPEN3D VISUALIZATION ---------
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(projected_lidar_points)
    pcd_full.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Full Lidar Point Cloud ({base_name})", width=800, height=600, left=50, top=50)
    vis.add_geometry(pcd_full)
    for line_set in line_sets:
        vis.add_geometry(line_set)
    opt = vis.get_render_option()
    opt.point_size = 2
    vis.run()
    vis.capture_screen_image(f"{detections_folder}/{base_name}_full_colored_with_boxes.png")
    vis.destroy_window()
    o3d.io.write_point_cloud(f"{detections_folder}/{base_name}_full_colored.ply", pcd_full)

    # Save all bounding boxes as a single LineSet
    if line_sets:
        all_lines = o3d.geometry.LineSet()
        all_points = []
        all_lines_idx = []
        all_colors = []
        point_offset = 0
        for ls in line_sets:
            pts = np.asarray(ls.points)
            lines = np.asarray(ls.lines) + point_offset
            cols = np.asarray(ls.colors)
            all_points.append(pts)
            all_lines_idx.append(lines)
            all_colors.append(cols)
            point_offset += pts.shape[0]
        all_points = np.vstack(all_points)
        all_lines_idx = np.vstack(all_lines_idx)
        all_colors = np.vstack(all_colors)
        all_lines.points = o3d.utility.Vector3dVector(all_points)
        all_lines.lines = o3d.utility.Vector2iVector(all_lines_idx)
        all_lines.colors = o3d.utility.Vector3dVector(all_colors)
        o3d.io.write_line_set(f"{detections_folder}/{base_name}_all_bboxes.ply", all_lines)

    # --------- EVALUATION ---------
    eval_rows = []
    for det_idx, indices in enumerate(lidar_indices_per_car_box):
        if len(indices) == 0 or det_idx not in det_idx_to_bbox_idx:
            continue
        bbox_idx = det_idx_to_bbox_idx[det_idx]
        car = bbox_data[bbox_idx]
        corners = np.array(car["corners_cam0"], dtype=np.float32)
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))
        # Map indices to masked indices for evaluation
        masked_indices = [orig_idx_to_masked_idx[idx] for idx in indices if idx in orig_idx_to_masked_idx]
        car_points = projected_lidar_points[masked_indices]
        inside_mask = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(car_points))
        num_inside = len(inside_mask)
        num_total = len(car_points)
        percent_inside = num_inside / num_total if num_total > 0 else 0
        eval_rows.append({
            "image": base_name,
            "car_idx": det_idx,
            "bbox_idx": bbox_idx,
            "points_inside": num_inside,
            "points_total": num_total,
            "percent_inside": percent_inside,
            "correct": percent_inside > 0.5
        })
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(f"{detections_folder}/{base_name}_evaluation.csv", index=False)
    all_eval_rows.extend(eval_rows)

# --- After all images, save summary table and plots ---
if all_eval_rows:
    all_eval_df = pd.DataFrame(all_eval_rows)
    all_eval_df.to_csv(f"{detections_folder}/all_images_evaluation.csv", index=False)

    # Bar chart: percent_inside for each detection (all images)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(all_eval_df)), all_eval_df['percent_inside'] * 100, color=['green' if c else 'red' for c in all_eval_df['correct']])
    plt.xlabel('Detected Car Index (across all images)')
    plt.ylabel('Percentage of Points Inside 3D BBox (%)')
    plt.title('Detection Quality: Percentage of Points Inside 3D Bounding Box')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{detections_folder}/all_images_evaluation_bar_chart.png")
    plt.close()

    # Per-image average percentage bar chart
    avg_percent_per_image = all_eval_df.groupby('image')['percent_inside'].mean() * 100
    plt.figure(figsize=(12, 6))
    avg_percent_per_image.plot(kind='bar', color='skyblue')
    plt.xlabel('Image')
    plt.ylabel('Average % of Points Inside 3D BBoxes')
    plt.title('Average Detection Quality per Image')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{detections_folder}/per_image_average_evaluation_bar_chart.png")
    plt.close()

    # CSV and Pie Chart for Detection Quality Summary
    total_cars = len(all_eval_df)
    correct_cars = (all_eval_df['percent_inside'] > 0.5).sum()
    incorrect_cars = total_cars - correct_cars
    overall_percentage = (correct_cars / total_cars) * 100 if total_cars > 0 else 0

    summary_df = pd.DataFrame({
        'total_cars': [total_cars],
        'correct_cars_above_50%': [correct_cars],
        'incorrect_cars_below_or_equal_50%': [incorrect_cars],
        'overall_correct_percentage': [overall_percentage]
    })
    summary_df.to_csv(f"{detections_folder}/detection_quality_summary.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.pie(
        [correct_cars, incorrect_cars],
        labels=['Correct (>50%)', 'Incorrect (≤50%)'],
        autopct='%1.1f%%',
        colors=['green', 'red'],
        startangle=90
    )
    plt.title('Overall Detection Quality (All Images)')
    plt.tight_layout()
    plt.savefig(f"{detections_folder}/detection_quality_pie_chart.png")
    plt.close()