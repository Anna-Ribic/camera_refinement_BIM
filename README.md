# Camera Refinement BIM Helper Functions

This repository contains helper functions for camera refinement with Building Information Modeling (BIM) models with the FACaP (https://github.com/Anna-Ribic/FACaP) pipeline. In order to use these functionalities you need to download the [ConSLAM](https://github.com/mac137/ConSLAM) dataset as outlined in their repository.

## Helper Functions

### Dense Depth from Lidar
- **File:** `projections/dense_depth_from_lidar.py`
- **Description:** Generates a dense depth map from sparse lidar data using interpolation.

### Dense Depth with CompletionFormer
- **File:** `projections/CompletionFormer`
- **Description:** Generates a dense depth map from sparse lidar data or another dense depth image using [CompletionFormer](https://github.com/youmi-zym/CompletionFormer). Clone the CompletionFormer repo and place iterate.sh and runnet.py in the 'src' folder. Running the iterate.sh script runs the CompletionFormer pipeline for all images in the specified folder path.

### Create Synthetic Trajectory
- **File:** `trajectories/model_drift.py`
- **Description:** Generates a .txt file specifying offset trajectory from groundtruth. Groundtruth camera poses for ConSLAM can be found [here](https://drive.google.com/file/d/1RWDW8mFWtAcG4TkiAherr0B9nyX6v3Qu/view?usp=sharing).

### Create Scan for FACaP 
- **File:** `projections/create_scan.py`
- **Description:** Creates a Scan folder from a trajectory file. Depth maps can be found [here](https://drive.google.com/file/d/1CfH5QlGJp0ue0493MT-ZGwMn5pXSSnbX/view?usp=sharing) and [here](https://drive.google.com/file/d/1N7UuIEnn3ugclh8xIf5gKJksClSmVh3W/view?usp=drive_link). 

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

