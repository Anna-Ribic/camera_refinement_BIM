import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import open3d as o3d


# from pypcd import pypcd

def read_calibration_configuration(yaml_fname):
    path2file = Path(yaml_fname)
    if path2file.is_file():
        with open(yaml_fname, "r") as file_handle:
            calib_data = yaml.safe_load(file_handle)
        # Parse
        camera_info_msg = {}
        camera_info_msg['width'] = calib_data["image_width"]
        camera_info_msg['height '] = calib_data["image_height"]
        camera_info_msg['K'] = calib_data["camera_matrix"]["data"]
        camera_info_msg['D '] = calib_data["distortion_coefficients"]["data"]
        # camera_info_msg.R = calib_data["rectification_matrix"]["data"]
        # camera_info_msg.P = calib_data["projection_matrix"]["data"]
        # camera_info_msg.distortion_model = calib_data["camera_model"]
        # camera_info_msg['header'].frame_id = calib_data["camera_name"]
        return camera_info_msg
    else:
        text = "Lidar2Cam Projector: path to the cam calib file is not valid"
        raise RuntimeError(text)


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


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    rotation_matrix = np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])
    return rotation_matrix


def sample_depth_values_and_coordinates(depth_image, num_samples=1000):
    height, width = depth_image.shape[:2]

    # Generate random pixel coordinates
    random_u = np.random.randint(0, width, num_samples)
    random_v = np.random.randint(0, height, num_samples)

    # Sample depth values at the random coordinates
    depth_values = np.array([depth_image[v, u] for u, v in zip(random_u, random_v)])

    depth_mask = depth_values < 1.5*depth_image.mean()

    return depth_values[depth_mask], random_u[depth_mask], random_v[depth_mask]


def read_transformation_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = [list(map(float, line.split())) for line in lines]
    transformation_matrix = np.array(matrix)

    return transformation_matrix


if __name__ == "__main__":
    root = "../../ConSLAM/data/"
    image_dir = os.path.join(root, "rgb", "rgb_2058x1533")
    concat_dir = os.path.join(root, "rgb", "rgb_concat-2")
    velodyne_dir = os.path.join(root, "lidar", "lidar-pcd")
    calib_dir = os.path.join(root, "calib", "data_calib")
    result_dir = os.path.join(root, "lidar", "depth-np")
    sem_dir = os.path.join(root, "rgb", "sem_masks")

    output_dir = os.path.join("..", "output")

    all_points = []

    with open("../trajectories/offset_trajectory_0.25_2.5.txt", 'r') as file:
        next(file)
        f = 1
        for line in file:
            # Split the line into timestamp, translation, and rotation parts
            parts = line.split()
            timestamp = int(parts[0])
            translation = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rotation = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])

            print(timestamp)

            # Store transformation matrix
            transformation_matrix = create_transformation_matrix(translation, rotation)

            transformation_matrix_og = read_transformation_matrix(os.path.join(root, "normalized", "%16d.txt" % timestamp))

            print(transformation_matrix, transformation_matrix_og)

            # Load RGB and depth images
            if not os.path.exists(os.path.join(result_dir, "%06d.npy" % timestamp)):
                continue
            img = cv2.imread(os.path.join(image_dir, "%16d.png" % timestamp))

            depth_path = os.path.join(result_dir, "%06d.npy" % timestamp)
            depth = np.load(depth_path)#cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)/256

            assert img.shape[:-1] == depth.shape

            zs, us, vs = sample_depth_values_and_coordinates(depth, num_samples=5000)

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

            all_points.append(points_in_world)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis = 0))
        o3d.io.write_point_cloud(os.path.join(output_dir, 'pcfull_off.ply'), cloud)

            #with open(os.path.join(output_dir, 'pcfull.obj'), 'a') as ob:
            #    for i in points_in_world:
            #        #print('v ' + str(i[0].item()) + ' ' + str(i[1].item()) + ' ' + str(i[2].item()) )
            #        ob.write('v ' + str(i[0].item()) + ' ' + str(i[1].item()) + ' ' + str(i[2].item()))
            #        ob.write('\n')
            #f += 1


