# Evaluation of a 3D Object Detector

The goal is to understand how well a 3D object detector performs when applied to real sensor data. We focus on cars as the object of interest and use both **camera images** and **lidar point clouds** to evaluate detection quality.

---

## Project Overview

1. **Dataset**  
   - A reduced subset of the **KITTI-360 dataset** is used.  
   - It includes:
     - Camera images (left camera, `image_00`)  
     - Lidar point clouds (`velodyne_points`)  
     - Calibration files (lidar ↔ camera transformations)  
     - Pre-calculated 3D bounding boxes for cars (ground truth).  

2. **Detection Method**  
   - We use **YOLO (Ultralytics)** for semantic instance segmentation.  
   - YOLO provides pixel-wise masks of detected cars in camera images.  
   - These masks are projected into the 3D lidar point cloud.  
   - Each car gets its own **sub-pointcloud**, essentially creating a 3D object detector.  

3. **Evaluation**  
   - Ground truth 3D bounding boxes are used to check detection quality.  
   - Correct points = inside the bounding box.  
   - Incorrect points = outside the bounding box.  
   - Metrics: number of correct vs. incorrect points per detection.  

---

## Steps Followed

1. **Setup the environment** 

2. **Run YOLO segmentation**  
   - Apply YOLO segmentation on camera images.  
   - Keep only detections of class `car`.

3. **Project lidar points**  
   - Transform lidar points into the camera coordinate system using calibration files.  
   - Overlay points on the image and assign them to cars using YOLO masks.

4. **Build the 3D object detector**  
   - Separate the lidar cloud into sub-clouds, one per detected car.  
   - Visualize results with color-coded sub-pointclouds and 3D bounding boxes.

5. **Evaluate performance**  
   - Compare detected sub-pointclouds with ground truth bounding boxes.  
   - Count correct and incorrect points.  
   - Discuss results: is the detector reliable enough for real-world use?

---

## Next Steps

- Try larger datasets like full **KITTI** or **Astyx**.  
- Experiment with different object detectors.  
- Explore sensor fusion strategies (low-level, feature-level, high-level fusion).

