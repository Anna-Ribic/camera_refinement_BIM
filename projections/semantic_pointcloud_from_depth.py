import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import open3d as o3d



# From LiDAR coordinate system to Camera Coordinate system
def lidar2cam(pts_3d_lidar, L2C, R0):
    n = pts_3d_lidar.shape[0]
    pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n, 1))))
    pts_3d_cam_ref = np.dot(pts_3d_hom, np.transpose(L2C))
    pts_3d_cam_rec = np.transpose(np.dot(R0, np.transpose(pts_3d_cam_ref)))
    return pts_3d_cam_rec


# From Camera Coordinate system to Image frame
def rect2Img(rect_pts, img_width, img_height, P):
    n = rect_pts.shape[0]
    points_hom = rect_pts  # np.hstack((rect_pts, np.ones((n, 1))))
    points_2d = np.dot(points_hom, np.transpose(P))  # nx3
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]

    mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] <= img_width) & (points_2d[:, 1] >= 0) & (
            points_2d[:, 1] <= img_height)
    mask = mask & (rect_pts[:, 2] > 2)
    return points_2d[mask, 0:2], mask

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


def read_transformation_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = [list(map(float, line.split())) for line in lines]
    transformation_matrix = np.array(matrix)

    return transformation_matrix

def sample_depth_values_and_coordinates_grid(depth_image, grid_size=10):
    # Get the height and width of the depth image
    height, width = depth_image.shape[:2]

    # Create a grid of pixel coordinates
    grid_u, grid_v = np.meshgrid(np.linspace(0, width - 1, grid_size),
                                 np.linspace(0, height - 1, grid_size))

    # Flatten the grid coordinates
    grid_u_flat = grid_u.flatten().astype(int)
    grid_v_flat = grid_v.flatten().astype(int)

    # Sample depth values at the grid coordinates
    depth_values = np.array([depth_image[v, u] for u, v in zip(grid_u_flat, grid_v_flat)])

    # Create a depth mask based on a condition (in this case, less than 1.5 times the mean depth)
    depth_mask = depth_values < 1.5 * depth_image.mean()

    # Filter depth values and corresponding coordinates based on the depth mask
    filtered_depth_values = depth_values[depth_mask]
    filtered_grid_u = grid_u_flat[depth_mask]
    filtered_grid_v = grid_v_flat[depth_mask]

    return filtered_depth_values, filtered_grid_u, filtered_grid_v


if __name__ == "__main__":
    root = "../data/"
    image_dir = os.path.join(root, "rgb", "rgb_2058x1533")
    concat_dir = os.path.join(root, "rgb", "rgb_concat-2")
    velodyne_dir = os.path.join(root, "lidar", "lidar-pcd")
    calib_dir = os.path.join(root, "calib", "data_calib")
    result_dir = os.path.join(root, "lidar", "depth-np")
    sem_dir = os.path.join('../../ConSLAM/data/', "rgb", "sem_masks")


    with open('../data/groundtruth2.txt', 'r') as file:
        next(file)
        f = 1
        for line in file:
            # Split the line into timestamp, translation, and rotation parts
            line ="1650963727934978 1.584190000000000111e+02 8.418559999999999377e+01 1.201939999999999920e+01 -3.192495874041319237e-02 5.096048642650186263e-03 4.248283997205117291e-01 9.046964452724262085e-01"
            parts = line.split()
            timestamp = int(parts[0])
            translation = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rotation = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])

            #timestamp = 1650963727934978

            # Store transformation matrix
            transformation_matrix = create_transformation_matrix(translation, rotation)

            #transformation_matrix = read_transformation_matrix(os.path.join(root, "normalized", "%16d.txt" % timestamp))

            # Load RGB and depth images
            print(transformation_matrix)

            print(os.path.join(image_dir, "%16d.png" % timestamp))
            print(os.path.join(sem_dir, "%16d.png" % timestamp))
            sem = cv2.imread(os.path.join(sem_dir, "%16d.png" % timestamp))

            depth_path = os.path.join(result_dir, "%06d.npy" % timestamp)
            depth = np.load(depth_path)#cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)/256

            #assert img.shape[:-1] == depth.shape

            zs, us, vs = sample_depth_values_and_coordinates_grid(depth, grid_size=200)#sample_depth_values_and_coordinates(depth)

            color_values = np.array([sem[v, u] for u, v in zip(us, vs)])
            print('sem',color_values.shape)

            fx = 2351.85909
            fy = 2353.56734
            cx = 1017.24032
            cy = 788.70774

            x_cam = zs * (us - cx) / fx
            y_cam = zs * (vs - cy) / fy

            hom = np.ones_like(x_cam.reshape(-1,1))
            #3d points in camera frame from depth maps
            points_in_cam_hom = np.stack((x_cam.reshape(-1,1), y_cam.reshape(-1,1), zs.reshape(-1,1), hom ), axis=1)

            projection4undist = np.identity(4)
            velo_to_cam = [[0.05758374, -0.99833897, -0.00184651, -0.0081025],
                           [-0.0017837, 0.0017467, -0.99999688, -0.0631593],
                           [0.99833909, 0.05758685, -0.00168015, -0.0215235],
                           [0.0, 0.0, 0.0, 1.0]]

            cam_to_velo = np.linalg.inv(velo_to_cam)

            points_in_velo = lidar2cam(points_in_cam_hom[:, 0:3].squeeze(), cam_to_velo, projection4undist)

            points_in_world = transformation_matrix @ points_in_velo.squeeze().T

            points_in_world = points_in_world.T[:, :-1]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points_in_world)
            cloud.colors = o3d.utility.Vector3dVector(color_values.astype(np.float) / 255.0)
            o3d.io.write_point_cloud(os.path.join('../output/sem.ply'), cloud)

            break





