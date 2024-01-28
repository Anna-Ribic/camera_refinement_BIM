import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import open3d as o3d
#from pypcd import pypcd

def dense_map(Pts, n, m, grid, ex=40):
    ng = 2 * grid + 1
    
    mX = np.zeros((m,n)) + float("inf")
    mY = np.zeros((m,n)) + float("inf")
    mD = np.zeros((m,n))
    mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]

    KmX = np.zeros((ng+ex, ng, m - ng-ex, n - ng))
    KmY = np.zeros((ng+ex, ng, m - ng-ex, n - ng))
    KmD = np.zeros((ng+ex, ng, m - ng-ex, n - ng))
    
    for i in range(ng+ex):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng-ex + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng-ex + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng-ex + i), j : (n - ng + j)]
    S = np.zeros_like(KmD[0,0])
    Y = np.zeros_like(KmD[0,0])
    
    for i in range(ng+ex):
        for j in range(ng):
            s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s
    
    S[S == 0] = 1
    out = np.zeros((m,n))
    out[(grid+int(ex/2)) + 1 : -(grid+int(ex/2)), grid + 1 : -grid] = Y/S

    for i in range(m):
        out[i, : grid +1 ] = np.full((1, grid+1),out[i, grid+1])
        out[i, -grid : ] = np.full((1, grid),out[i, -(grid+1)])

    for i in range(n):
        out[:(grid+int(ex/2)) +1, i ] = np.full(( (grid+int(ex/2))+1),out[(grid+int(ex/2)) +1, i])
        out[-(grid+int(ex/2)):, i] = np.full(((grid+int(ex/2))),out[-((grid+int(ex/2))+1), i])

    return out

def read_calibration_configuration(yaml_fname):

    path2file = Path(yaml_fname)
    if path2file.is_file():
        with open(yaml_fname, "r") as file_handle:
            calib_data = yaml.safe_load(file_handle)
        # Parse
        camera_info_msg = {}
        camera_info_msg['width'] = calib_data["image_width"]
        camera_info_msg['height ']= calib_data["image_height"]
        camera_info_msg['K'] = calib_data["camera_matrix"]["data"]
        camera_info_msg['D ']= calib_data["distortion_coefficients"]["data"]
        #camera_info_msg.R = calib_data["rectification_matrix"]["data"]
        #camera_info_msg.P = calib_data["projection_matrix"]["data"]
        #camera_info_msg.distortion_model = calib_data["camera_model"]
        #camera_info_msg['header'].frame_id = calib_data["camera_name"]
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
    points_hom = rect_pts #np.hstack((rect_pts, np.ones((n, 1))))
    points_2d = np.dot(points_hom, np.transpose(P))  # nx3
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]

    mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] <= img_width) & (points_2d[:, 1] >= 0) & (
                points_2d[:, 1] <= img_height)
    mask = mask & (rect_pts[:, 2] > 2)
    return points_2d[mask, 0:2], mask


if __name__ == "__main__":
    root = "data/"
    image_dir = os.path.join(root, "rgb", "rgb_2058x1533")
    concat_dir = os.path.join(root, "rgb", "rgb_concat-2")
    velodyne_dir = os.path.join(root, "lidar", "lidar-pcd")
    calib_dir = os.path.join(root, "calib", "data_calib")
    result_dir = os.path.join(root, "lidar", "depth-np")

    for images in os.listdir(velodyne_dir):

        # check if the image ends with png
        if (scan.endswith(".pcd")):

            print(scan)
            # Data id
            cur_id = int(scan[:-4]) #1650963683657809
            # Loading the image
            img = cv2.imread(os.path.join(image_dir, "%16d.png" % cur_id))
            # Loading the LiDAR data
            pcd = o3d.io.read_point_cloud(os.path.join(velodyne_dir, "%06d.pcd" % cur_id))
            lidar = out_arr = np.asarray(pcd.points)

            # Loading Calibration
            calib = read_calibration_configuration(os.path.join(calib_dir, 'calib_rgb.yaml'))
            cam_mtx = np.reshape(calib['K'], newshape=(3, 3))
            cam_mtx=np.c_[cam_mtx, np.zeros(3)]  # add a column
            projection4undist = np.identity(4)
            velo_to_cam = [[0.05758374, -0.99833897, -0.00184651, -0.0081025],
                            [-0.0017837,   0.0017467,  -0.99999688, -0.0631593],
                            [0.99833909,  0.05758685, -0.00168015, -0.0215235],
                            [0.0, 0.0, 0.0, 1.0]]

            # From LiDAR coordinate system to Camera Coordinate system
            lidar_rect = lidar2cam(lidar[:, 0:3], velo_to_cam, projection4undist)
            # From Camera Coordinate system to Image frame
            lidarOnImage, mask = rect2Img(lidar_rect, img.shape[1], img.shape[0], cam_mtx)
            # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
            lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)

            factor = 0.5
            img = cv2.resize(img, (0, 0), fx=factor, fy=factor)
            lidarOnImage[:, 0] *= factor
            lidarOnImage[:, 1] *= factor

            grid = 5
            ex = 80
            out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], grid, ex)
            out = cv2.resize(out, (0, 0), fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_LINEAR)

            np.save(os.path.join(result_dir, "%06d.npy" % cur_id), out)

