import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import open3d as o3d
import decimal



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
    depth_mask = depth_values < depth_image.mean()

    # Filter depth values and corresponding coordinates based on the depth mask
    filtered_depth_values = depth_values[depth_mask]
    filtered_grid_u = grid_u_flat[depth_mask]
    filtered_grid_v = grid_v_flat[depth_mask]

    return filtered_depth_values, filtered_grid_u, filtered_grid_v


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

fx = 2351.85909
fy = 2353.56734
cx = 1017.24032
cy = 788.70774

velo_to_cam = [[0.05758374, -0.99833897, -0.00184651, -0.0081025],
               [-0.0017837, 0.0017467, -0.99999688, -0.0631593],
               [0.99833909, 0.05758685, -0.00168015, -0.0215235],
               [0.0, 0.0, 0.0, 1.0]]

cam_to_velo = np.linalg.inv(velo_to_cam)

if __name__ == "__main__":
    root = "../data/"
    image_dir = os.path.join(root, "rgb", "rgb_2058x1533")
    depth_dir = os.path.join(root, "lidar", 'depth-np') #"lidar-cf-np")
    sem_dir = os.path.join(root, "rgb", "masks")
    velodyne_dir = os.path.join('../../ConSLAM/data/', "lidar", "lidar-pcd")


    pose_file_path ='../trajectories/semantic_groundtruth.txt'

    sem_list = os.listdir(sem_dir)

    all_points = []
    all_colors = []

    with open(pose_file_path, 'r') as file:
        f = 1
        for line in file:
            # Split the line into timestamp, translation, and rotation parts
            parts = line.split()
            timestamp = int(decimal.Decimal(parts[0])*1000000)

            #Find matching file which is close to timestamp
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
            #transformation_matrix = read_transformation_matrix(os.path.join(root, "normalized", "%16d.txt" % timestamp))

            # Load semantic and depth images
            sem = cv2.imread(os.path.join(sem_dir, matching_file))

            if not os.path.exists(os.path.join(image_dir, matching_file)):
                print(os.path.join(image_dir, matching_file), ' doesnt exist.')
                continue
            rgb = cv2.imread(os.path.join(image_dir, matching_file))


            ##### Load and erode columns masks
            h, w, _ = sem.shape

            column = np.zeros(sem.shape[:2])
            mask = np.all(sem == np.asarray([227.,71.,223.]), axis=-1)
            column[mask] = 1

            kernel = np.ones((5, 5), np.uint8)  # 5x5 square kernel

            # Perform erosion
            eroded_image = cv2.erode(column, kernel, iterations=20)

            sem[eroded_image == 1] = np.asarray([230., 71., 223.])

            ########

            ####Load Depth data

            depth_path = os.path.join(depth_dir, "%06d.npy" % matching_file_timestamp)
            if not os.path.exists(depth_path):
                print(depth_path, ' doesnt exist.')
                continue
            depth = np.load(depth_path)

            depth = cv2.resize(depth, ( 2064,1544), interpolation=cv2.INTER_LINEAR)
            if depth.max() == 0:
                print('zero depth image detected')
                continue

            # Smooth ceiling and floor depth
            mask = np.all(sem == np.asarray([51., 153., 204.]), axis=-1).astype(np.uint8)
            blur_kernel_size = 25
            depth = smooth_depth_in_mask(depth, mask, blur_kernel_size)

            mask = np.all(sem == np.asarray([250., 183., 50.]), axis=-1).astype(np.uint8)
            blur_kernel_size = 25
            depth = smooth_depth_in_mask(depth, mask, blur_kernel_size)

            #Perform projection
            zs, us, vs = sample_depth_values_and_coordinates_grid(depth, grid_size=400)

            color_values = np.array([sem[v, u] for u, v in zip(us, vs)])

            x_cam = zs * (us - cx) / fx
            y_cam = zs * (vs - cy) / fy

            hom = np.ones_like(x_cam.reshape(-1,1))
            #3d points in camera frame from depth maps
            points_in_cam_hom = np.stack((x_cam.reshape(-1,1), y_cam.reshape(-1,1), zs.reshape(-1,1), hom ), axis=1)

            projection4undist = np.identity(4)

            points_in_velo = lidar2cam(points_in_cam_hom[:, 0:3].squeeze(), cam_to_velo, projection4undist)

            points_in_world = transformation_matrix @ points_in_velo.squeeze().T

            points_in_world = points_in_world.T[:, :-1]

            all_points.append(points_in_world)

            all_colors.append(color_values)

            f += 1


        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0).astype(float)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(all_points)
        cloud.colors = o3d.utility.Vector3dVector(all_colors / 255.0)
        o3d.io.write_point_cloud(os.path.join('../output/sem_full.ply'), cloud)

        # Extract only points with 4 main categories
        masks = [(all_colors == np.asarray(color_val)).all(axis=1) for color_val in segments.values()]
        # Combine the masks using logical OR
        final_mask = np.any(masks, axis=0)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(all_points[final_mask])
        cloud.colors = o3d.utility.Vector3dVector(all_colors[final_mask] / 255.0)
        o3d.io.write_point_cloud(os.path.join('../output/sem.ply'), cloud)


        #save segment pointclouds
        for seg, color in segments.items():
            mask = all_colors == np.asarray(color)
            mask = np.asarray([all(row) for row in mask])
            masked_points = all_points[mask]
            masked_colors = all_colors[mask]
            if seg == 'columns':
                column_mask = masked_points[:,2] < np.mean(masked_points[:,2])
                masked_points = masked_points[column_mask]
                masked_colors = masked_colors[column_mask]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(masked_points)
            cloud.colors = o3d.utility.Vector3dVector(masked_colors / 255.0)
            o3d.io.write_point_cloud(os.path.join('../output/sem_' + seg + '.ply'), cloud)


        print('finished writing point cloud')





