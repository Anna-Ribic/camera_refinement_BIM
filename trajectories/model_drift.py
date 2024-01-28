import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import open3d as o3d
import torch


def create_transformation_matrix(translation, rotation):
    # Assuming rotation is in quaternion form
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation
    return transformation_matrix


def sample_depth_values_and_coordinates(depth_image, num_samples=1000):
    height, width = depth_image.shape[:2]

    # Generate random pixel coordinates
    random_u = np.random.randint(0, width, num_samples)
    random_v = np.random.randint(0, height, num_samples)

    # Sample depth values at the random coordinates
    depth_values = [depth_image[v, u] for u, v in zip(random_u, random_v)]

    return depth_values, random_u, random_v


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    rotation_matrix = np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])
    return rotation_matrix

def rotation_matrix_to_quaternion(rot_matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Args:
    - rot_matrix (np.ndarray): 3x3 rotation matrix.

    Returns:
    - quaternion (np.ndarray): Quaternion [qx, qy, qz, qw].
    """
    # Ensure the input is a NumPy array
    rot_matrix = np.array(rot_matrix)

    # Extract the trace and diagonal elements of the rotation matrix
    trace = np.trace(rot_matrix)
    r = np.sqrt(1 + trace)

    qw = 0.5 * r
    qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) / (2 * r)
    qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) / (2 * r)
    qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) / (2 * r)

    quaternion = np.array([qx, qy, qz, qw])

    return quaternion


def read_transformation_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = [list(map(float, line.split())) for line in lines]
    transformation_matrix = np.array(matrix)

    return transformation_matrix

def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )

def rot_theta(th):
    return torch.tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


if __name__ == "__main__":
    root = "../data/"
    image_dir = os.path.join(root, "rgb", "rgb_2058x1533")
    concat_dir = os.path.join(root, "rgb", "rgb_concat-2")
    velodyne_dir = os.path.join(root, "lidar", "lidar-pcd")
    calib_dir = os.path.join(root, "calib", "data_calib")
    result_dir = os.path.join(root, "lidar", "depth-np")
    sem_dir = os.path.join(root, "rgb", "sem_masks")

    dev_trans = 0.25
    dev_rot = 2.5

    with open('offset_trajectory_'+str(dev_trans)+'_'+str(dev_rot)+'.txt', 'a') as ob:
        ob.write("# timestamp tx ty tz qx qy qz qw\n")

    with open('groundtruth2.txt', 'r') as file:
        next(file)
        f = 1
        prev = np.asarray([0,0,0], dtype=np.float32)
        for line in file:
            # Split the line into timestamp, translation, and rotation parts
            parts = line.split()
            timestamp = int(parts[0])
            translation = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rotation = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])

            # Store transformation matrix
            #transformation_matrix = create_transformation_matrix(translation, rotation)

            transformation_matrix = read_transformation_matrix(os.path.join(root, "normalized", "%16d.txt" % timestamp))

            point = np.zeros((4, 1))
            point[-1] = 1
            cam = transformation_matrix @ point
            cam = cam[:-1].squeeze()

            phi = np.random.normal(0, dev_rot)
            theta = np.random.normal(0, dev_rot)

            rot_mat = rot_theta(theta / 180.0 * np.pi) @ rot_phi(phi/ 180.0 * np.pi) @ transformation_matrix
            rot_quat = rotation_matrix_to_quaternion(rot_mat)

            offset = np.random.normal(prev,  np.asarray([dev_trans, dev_trans, 0.1]))
            prev = offset

            cam = cam + offset

            with open('offset_trajectory_'+str(dev_trans)+'_'+str(dev_rot)+'.txt', 'a') as ob:
                ob.write(parts[0]+' ' + str(cam[0].item()) + ' ' + str(cam[1].item()) + ' ' + str(cam[2].item()) + ' ' + str(rot_quat[0].item()) + ' ' + str(rot_quat[1].item()) + ' ' + str(rot_quat[2]) + ' ' + str(rot_quat[3].item()))
                ob.write('\n')






