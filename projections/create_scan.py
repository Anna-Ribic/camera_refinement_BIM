import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import open3d as o3d
import decimal




def create_transformation_matrix(translation, rotation):
    # Assuming rotation is in quaternion form
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation
    return transformation_matrix


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    rotation_matrix = np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])
    return rotation_matrix

def smooth_depth_in_mask(depth_image, binary_mask, blur_kernel_size, num_iterations=50):
    # Create a copy of the depth image
    smoothed_depth_image = depth_image.copy()

    # Create a region of interest (ROI) using the binary mask
    roi = cv2.bitwise_and(depth_image, depth_image, mask=binary_mask)

    # Apply blur/smoothing only within the ROI
    for _ in range(num_iterations):
        blurred_roi = cv2.GaussianBlur(roi, (3,  blur_kernel_size), 0)
        #blurred_roi = cv2.blur(roi, (blur_kernel_size, blur_kernel_size))
        roi = blurred_roi

    # Replace the ROI in the smoothed depth image
    smoothed_depth_image[binary_mask.astype(bool)] = blurred_roi[binary_mask.astype(bool)]

    return smoothed_depth_image

segments = {
    'floor': [51., 153., 204.],  
    'columns': [230., 71., 223.],
    'ceiling': [250., 183., 50.],
    'wall': [55., 250., 250.]
}


if __name__ == "__main__":
    root = "../data/"
    image_dir = os.path.join(root, "rgb", "rgb_2058x1533")
    concat_dir = os.path.join(root, "rgb", "rgb_concat-2")
    velodyne_dir = os.path.join(root, "lidar", "lidar-pcd")
    calib_dir = os.path.join(root, "calib", "data_calib")
    result_dir = os.path.join(root, "lidar", "depth-np")
    sem_dir = os.path.join(root, "rgb", "masks")

    #Modify this for different scans
    scan_dir = '../scan_lowres_002_500/'
    poses_txt = '../trajectories/semantic_groundtruth.txt' #'../trajectories/offset_trajectory_0.02_5.txt'

    os.makedirs(scan_dir, exist_ok=True)
    os.makedirs(os.path.join(scan_dir, 'arcore'), exist_ok=True)
    os.makedirs(os.path.join(scan_dir, 'segmentation'), exist_ok=True)

    sem_list = os.listdir(sem_dir)

    with open(poses_txt, 'r') as file:
        f = 1
        for line in file:
            # Split the line into timestamp, translation, and rotation parts
            parts = line.split()
            timestamp = int(decimal.Decimal(parts[0])*1000000)

            matching_files = [file for file in sem_list if file.startswith(str(timestamp)[:13]) and file.endswith(".png")]
            if not matching_files:
                continue
            matching_file = matching_files[0]
            matching_file_timestamp = int(matching_file[:-4])

            print(timestamp, matching_file)

            translation = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rotation = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])

            # Store transformation matrix
            transformation_matrix = create_transformation_matrix(translation, rotation)
            pose_filename = f"{scan_dir}/arcore/pose-{f:04d}.txt"
            np.savetxt(pose_filename, transformation_matrix)

            # Load RGB and depth images

            sem = cv2.imread(os.path.join(sem_dir, matching_file))
            sem = cv2.resize(sem, (516, 386), interpolation=cv2.INTER_LINEAR)

            if not os.path.exists(os.path.join(image_dir, matching_file)):
                print(os.path.join(image_dir, matching_file), ' doesnt exist.')
                continue

            depth_path = os.path.join(result_dir, "%06d.npy" % matching_file_timestamp)
            if not os.path.exists(depth_path):
                print(depth_path, ' doesnt exist.')
                continue
            depth = np.load(depth_path)  # cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)/256

            depth = cv2.resize(depth, (516, 386), interpolation=cv2.INTER_LINEAR) #(2064, 1544)
            if depth.max() == 0:
                print('Zero-depth image detected.')
                continue

            rgb = cv2.imread(os.path.join(image_dir, matching_file))
            rgb = cv2.resize(rgb, (516, 386), interpolation=cv2.INTER_LINEAR)
            rgb_filename = f"{scan_dir}/arcore/frame-{f:04d}.png"
            cv2.imwrite(rgb_filename, rgb)

            #####Load and erode sem mask
            h, w, _ = sem.shape

            column = np.zeros(sem.shape[:2])
            mask = np.all(sem == np.asarray([227.,71.,223.]), axis=-1)
            column[mask] = 1

            kernel = np.ones((5, 5), np.uint8)  # 5x5 square kernel

            # Perform erosion
            eroded_image = cv2.erode(column, kernel, iterations=20)

            sem[eroded_image == 1] = np.asarray([230., 71., 223.])

            ########

            ###Save Wall, Column and Floor Mask
            for seg, color in segments.items():
                mask = np.all(sem == np.asarray(color), axis=-1)
                mask_filename = f"{scan_dir}/segmentation/frame-{f:04d}_{seg}.png"
                cv2.imwrite(mask_filename, mask.astype(np.uint8)*255)

            ####Load Depth data


            mask = np.all(sem == np.asarray([51., 153., 204.]), axis=-1).astype(np.uint8)
            blur_kernel_size = 25
            depth = smooth_depth_in_mask(depth, mask, blur_kernel_size)

            mask = np.all(sem == np.asarray([250., 183., 50.]), axis=-1).astype(np.uint8)
            blur_kernel_size = 25
            depth = smooth_depth_in_mask(depth, mask, blur_kernel_size)

            depth_filename= f"{scan_dir}/arcore/depth-{f:04d}.npy"
            np.save(depth_filename, depth)

            f += 1






