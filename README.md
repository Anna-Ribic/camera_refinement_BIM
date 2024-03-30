# Camera Refinement BIM Helper Functions

This repository contains helper functions for camera refinement with Building Information Modeling (BIM) models with the FACaP (https://github.com/Anna-Ribic/FACaP) pipeline.

## Helper Functions

### Dense Depth from Lidar
- **File:** `projections/dense_depth_from_lidar.py`
- **Description:** Generates a dense depth map from sparse lidar data using interpolation.

### Create Synthetic Trajectory
- **File:** `trajectories/model_drift.py`
- **Description:** Generates a .txt file specifying offset trajectory from groundtruth

### Create Scan for FACaP 
- **File:** `projections/create_scan.py`
- **Description:** Creates a Scan folder from a trajectory file.

- **File:** `projections/floorplan_from_json.py`
- **Description:** Generates a `floorplan.npy` file from a JSON file.

### Example Scan
- **File:** `scan_lowres/`
- **Description:** Example Scan for the ConSLAM dataset (https://github.com/mac137/ConSLAM).

### Semantic Pointcloud from Depth
- **File:** `projections/semantic_pointcloud_from_depth.py`
- **Description:** Creates a `.ply` file from semantic masks and depth data.

### Evaluation
- **File:** `trajectories/map_metrics.py`
- **Description:** Compute MME, MPV, MOM for pointcloud

- **File:** `trajectories/ate.py`
- **Description:** Compute ATE RMSE and ROT RMSE for estimated and groundtruth trajectory

### Visualization
- **File:** `notebooks/Optimization_Vis.ipynb`
- **Description:** Visualize optimization behaviour from FACaP log file

