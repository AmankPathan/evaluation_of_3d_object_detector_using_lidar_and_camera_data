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

## Discussion

- In 3D object detection with Lidar data, the absolute number of points inside a bounding box might seems accurate but it is not always valid. Because that number can be affected with some external factors such as distance to object, occlusion, lidar density. That is why we do not consider absolute count as a parameter for evaluating our result. Instead we use relative percentage that simply tells us how many points fall inside the box, we then calculate the percentage of the objects total points that are captured. This gives us a normalized, scene-independent measure of localization accuracy.
- **Does this solution actually work?** After evaluation we get clear idea of how well the detector performs in some conditions. It successfully matches the data with cars
and measures how correctly the system identifies where the object(car) is located. The percentage based scoring system make it easy to compare performance across given scenarios. But there are some challenges which we need to be consider:
*Detection and Segmentation*: The accuracy depends heavily on the quality of 2D detections and segmentation masks. Missed detections, poor segmentation can lead to bad results.
*Bounding Box Matching*: Box matching may fail in dense or complex environments.
*Real-time Use*: The system is not designed to handle real-time performance. This can only be used for research or study purpose.
- **Would I put this in an actual car?** Not yet. While it is good for research and development, it is not yet ready for real-world driving conditions. The system needs improvements in speed, reliability and error handling before putting it to an actual vehicle. For something critical like autonomous driving, we need multiple backup systems and a solid real-time performance which is similar to what we do multi-threading in Robot Operating System(ROS).

<img width="600" height="600" alt="detection_quality_pie_chart" src="https://github.com/user-attachments/assets/60a305ab-d825-484b-9e0b-a7e234660d96" />
<img width="1200" height="600" alt="all_images_evaluation_bar_chart" src="https://github.com/user-attachments/assets/160cb458-32ac-42fc-9f01-c38408e9bb5b" />

---

## Next Steps

- Try larger datasets like full **KITTI** or **Astyx**.  
- Experiment with different object detectors.  
- Explore sensor fusion strategies (low-level, feature-level, high-level fusion).

---



