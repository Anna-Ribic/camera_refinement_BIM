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
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return rotation_matrix


def sample_depth_values_and_coordinates(depth_image, num_samples=1000):
    height, width = depth_image.shape[:2]

    # Generate random pixel coordinates
    random_u = np.random.randint(0, width, num_samples)
    random_v = np.random.randint(0, height, num_samples)

    # Sample depth values at the random coordinates
    depth_values = np.array([depth_image[v, u] for u, v in zip(random_u, random_v)])

    return depth_values, random_u, random_v


def read_transformation_matrix(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = [list(map(float, line.split())) for line in lines]
    transformation_matrix = np.array(matrix)

    return transformation_matrix


if __name__ == "__main__":
    root = "data/"
    image_dir = os.path.join(root, "rgb", "rgb_2058x1533")
    concat_dir = os.path.join(root, "rgb", "rgb_concat-2")
    velodyne_dir = os.path.join(root, "lidar", "lidar-pcd")
    calib_dir = os.path.join(root, "calib", "data_calib")
    result_dir = os.path.join(root, "lidar", "lidar-depth-dense-2")

    with open('data/groundtruth.txt', 'r') as file:
        pass
        """next(file)
        f = 1
        for line in file:
            # Split the line into timestamp, translation, and rotation parts
            parts = line.split()
            timestamp = int(parts[0])
            translation = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rotation = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])

            # Store transformation matrix
            #transformation_matrix = create_transformation_matrix(translation, rotation)

            transformation_matrix = read_transformation_matrix(os.path.join(root, "normalized", "%16d.txt" % timestamp))

            # Load RGB and depth images
            img = cv2.imread(os.path.join(image_dir, "%16d.png" % timestamp))
            depth_path = os.path.join(result_dir, "depth_map_%06d.png" % timestamp)
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)/256
            print('meandepth', np.mean(depth))
            
            assert img.shape[:-1] == depth.shape

            zs, us, vs = sample_depth_values_and_coordinates(depth)

            fx = 2351.85909
            fy = 2353.56734
            cx = 1017.24032
            cy = 788.70774

            x_cam = zs * (us - cx) / fx
            y_cam = zs * (vs - cy) / fy

            hom = np.ones_like(x_cam.reshape(1000,1))
            #3d points in camera frame from depth maps
            points_in_cam_hom = np.stack((x_cam.reshape(1000,1), y_cam.reshape(1000,1), zs.reshape(1000,1), hom ), axis=1)


            ################################################

            calib = read_calibration_configuration(os.path.join(calib_dir, 'calib_rgb.yaml'))
            cam_mtx = np.reshape(calib['K'], newshape=(3, 3))
            cam_mtx = np.c_[cam_mtx, np.zeros(3)]
            print(cam_mtx)
            projection4undist = np.identity(4)
            velo_to_cam = [[0.05758374, -0.99833897, -0.00184651, -0.0081025],
                           [-0.0017837, 0.0017467, -0.99999688, -0.0631593],
                           [0.99833909, 0.05758685, -0.00168015, -0.0215235],
                           [0.0, 0.0, 0.0, 1.0]]

            cam_to_velo = np.linalg.inv(velo_to_cam)

            pcd = o3d.io.read_point_cloud(os.path.join(velodyne_dir, "%06d.pcd" % timestamp))
            # 3d points in lidar frame
            lidar = out_arr = np.asarray(pcd.points)
            print(lidar.shape)

            point = lidar[:5].reshape(-1, 3)
            print('point', point)

            lidar_rect = lidar2cam(point, velo_to_cam, projection4undist)

            print('point_in_cam',lidar_rect)

            lidarOnImage, mask = rect2Img(lidar_rect, img.shape[1], img.shape[0], cam_mtx)

            print('lidarOnImage', lidarOnImage)

            lidar_rect_back = lidar2cam(lidar_rect[:, 0:3], cam_to_velo, projection4undist)

            print('lidar_rect_back', lidar_rect_back)

            if lidarOnImage.shape[0] != 0:
                u , v= lidarOnImage[:, 0].astype(int), lidarOnImage[:, 1].astype(int)
                print('u, v', u, v)
                print(depth[v,u]

            factor = 0.5
            img = cv2.resize(rgb, (0, 0), fx=factor, fy=factor)
            lidarOnImage[:, 0] *= factor
            lidarOnImage[:, 1] *= factor"""




        """pcd = o3d.io.read_point_cloud(os.path.join(velodyne_dir, "%06d.pcd" % timestamp))
            #3d points in lidar frame
            lidar = out_arr = np.asarray(pcd.points)
            print(lidar.shape)
            lidar_pos = lidar[:, 2][lidar[:, 2] > 0]
            print('meandlidar', np.mean(lidar_pos))
            calib = read_calibration_configuration(os.path.join(calib_dir, 'calib_rgb.yaml'))
            cam_mtx = np.reshape(calib['K'], newshape=(3, 3))
            cam_mtx = np.c_[cam_mtx, np.zeros(3)]
            print(cam_mtx)
            projection4undist = np.identity(4)
            velo_to_cam = [[0.05758374, -0.99833897, -0.00184651, -0.0081025],
                           [-0.0017837, 0.0017467, -0.99999688, -0.0631593],
                           [0.99833909, 0.05758685, -0.00168015, -0.0215235],
                           [0.0, 0.0, 0.0, 1.0]]

            cam_to_velo = np.linalg.inv(velo_to_cam)

            # From LiDAR coordinate system to Camera Coordinate system
            lidar_rect = lidar2cam(lidar[:, 0:3], velo_to_cam, projection4undist)
            # From Camera Coordinate system to Image frame
            lidarOnImage, mask = rect2Img(lidar_rect, img.shape[1], img.shape[0], cam_mtx)
            lidar_rect_pos = lidar_rect[:, 2][lidar_rect[:, 2] > 0]
            print('meanlidarrect', np.mean(lidar_rect_pos))
            lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask, 2].reshape(-1, 1)), 1)
            print('meandlidaronimage', np.mean(lidarOnImage[:,2]))

            lidar_rect_back = lidar2cam(lidar_rect[:, 0:3], cam_to_velo, projection4undist)
            print('meandlidarback', np.mean(lidar_rect_back[:,2]))


            # 3d points in lidar from depth maps
            point_in_lidar = cam_to_velo @ points_in_cam_hom.squeeze().T

            #point_in_lidar = np.concatenate((point_in_lidar, np.ones((point_in_lidar.shape[0], 1))), axis=-1)

            #print(points_in_cam_hom.shape)
            #print(transformation_matrix.shape)
            #lidar = np.hstack((lidar, np.ones((lidar.shape[0], 1))))
            #print(lidar.shape)
            print(point_in_lidar.shape)
            points_in_world = transformation_matrix @ point_in_lidar #lidar.T #points_in_cam_hom

            print('here',points_in_world.squeeze())

            points_in_world = points_in_world.T[:, :-1]
            print(points_in_world.shape)

            with open('pc'+str(f)+'.obj', 'a') as ob:
                for i in points_in_world:
                    ob.write('v ' + str(i[0].item()) + ' ' + str(i[1].item()) + ' ' + str(i[2].item()))
                    ob.write('\n')

            f += 1"""


            #points_in_world = points_in_world[:, :-1].squeeze() / points_in_world[:, -1].repeat(3, axis=1)

    result_dir = os.path.join(root, "lidar", "depth-np")

    for images in os.listdir(velodyne_dir):

        # check if the image ends with png
        if (images.endswith(".pcd")):

            print(images)
            # Data id
            cur_id = int(images[:-4]) #1650963683657809
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

            # out = np.zeros((img.shape[0], img.shape[1]))
            # out[np.int32(lidarOnImage.T[1]), np.int32(lidarOnImage.T[0])] = lidarOnImage.T[2]

            grid = 5
            ex = 80
            out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], grid, ex)
            out = cv2.resize(out, (0, 0), fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_LINEAR)

            np.save(os.path.join(result_dir, "%06d.npy" % cur_id), out)



    #result_dir = os.path.join(root, "lidar", "lidar-depth-dense-2")

    """for images in os.listdir(result_dir):

            # check if the image ends with png
            #if (images.endswith(".png")):

                # Data id
                cur_id = int(images[16:-4])
                image_path = os.path.join(image_dir, images)
                depth_path = os.path.join(result_dir, "depth_map_%06d")

                print(cur_id)

                with open('depth.txt', 'a') as f:
                    f.write(str(cur_id)+' lidar/lidar-depth-dense-cf/' + images+'\n')
                if not os.path.isfile(os.path.join(result_dir, 'depth_map_%06d.png' % cur_id)):
                    print(images)
                    print(os.path.isfile(os.path.join(result_dir, 'depth_map_%06d.png' % cur_id))

                # Loading the image
                rgb = cv2.imread(os.path.join(image_dir, "%16d.png" % cur_id))
                # Loading the LiDAR data
                pcd = o3d.io.read_point_cloud(os.path.join(velodyne_dir, "%06d.pcd" % cur_id))
                lidar = out_arr = np.asarray(pcd.points)

                # Loading Calibration
                calib = read_calibration_configuration(os.path.join(calib_dir, 'calib_rgb.yaml'))
                cam_mtx = np.reshape(calib['K'], newshape=(3, 3))
                cam_mtx = np.c_[cam_mtx, np.zeros(3)]  # add a column
                projection4undist = np.identity(4)
                velo_to_cam = [[0.05758374, -0.99833897, -0.00184651, -0.0081025],
                               [-0.0017837, 0.0017467, -0.99999688, -0.0631593],
                               [0.99833909, 0.05758685, -0.00168015, -0.0215235],
                               [0.0, 0.0, 0.0, 1.0]]

                # From LiDAR coordinate system to Camera Coordinate system
                lidar_rect = lidar2cam(lidar[:, 0:3], velo_to_cam, projection4undist)
                # From Camera Coordinate system to Image frame
                lidarOnImage, mask = rect2Img(lidar_rect, rgb.shape[1], rgb.shape[0], cam_mtx)
                # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
                lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask, 2].reshape(-1, 1)), 1)

                factor = 0.5
                img = cv2.resize(rgb, (0,0), fx = factor, fy = factor)
                lidarOnImage[:, 0] *= factor
                lidarOnImage[:,1] *= factor

                #out = np.zeros((img.shape[0], img.shape[1]))
                #out[np.int32(lidarOnImage.T[1]), np.int32(lidarOnImage.T[0])] = lidarOnImage.T[2]

                grid = 7
                ex = 80
                out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], grid, ex)
                out = cv2.resize(out, (0,0), fx = 1/factor, fy = 1/factor, interpolation = cv2.INTER_LINEAR)
                #out = cv2.blur(out,(15,15))
                plt.figure(figsize=(20,40))
                plt.imsave(os.path.join(result_dir,"depth_map_%06d.png" % cur_id), out)
                heatmap = cv2.imread(os.path.join(result_dir,"depth_map_%06d.png" % cur_id))

                concat = cv2.hconcat([rgb, heatmap])
                cv2.imwrite(os.path.join(concat_dir,"depth_map_%06d.png" % cur_id), concat)"""
                #cv2.imwrite(os.path.join("depth_map_%06d.png" % cur_id), 10 *out)

    """result_dir = os.path.join(root, "lidar", "test")

    cur_id = 1650963758997008

    # Loading the image
    rgb = cv2.imread(os.path.join(image_dir, "%16d.png" % cur_id))
    # Loading the LiDAR data
    pcd = o3d.io.read_point_cloud(os.path.join(velodyne_dir, "%06d.pcd" % cur_id))
    lidar = out_arr = np.asarray(pcd.points)

    # Loading Calibration
    calib = read_calibration_configuration(os.path.join(calib_dir, 'calib_rgb.yaml'))
    cam_mtx = np.reshape(calib['K'], newshape=(3, 3))
    cam_mtx = np.c_[cam_mtx, np.zeros(3)]  # add a column
    projection4undist = np.identity(4)
    velo_to_cam = [[0.05758374, -0.99833897, -0.00184651, -0.0081025],
                   [-0.0017837, 0.0017467, -0.99999688, -0.0631593],
                   [0.99833909, 0.05758685, -0.00168015, -0.0215235],
                   [0.0, 0.0, 0.0, 1.0]]

    # From LiDAR coordinate system to Camera Coordinate system
    lidar_rect = lidar2cam(lidar[:, 0:3], velo_to_cam, projection4undist)
    print(lidar_rect)
    # From Camera Coordinate system to Image frame
    lidarOnImage, mask = rect2Img(lidar_rect, rgb.shape[1], rgb.shape[0], cam_mtx)
    # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
    lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask, 2].reshape(-1, 1)), 1)

    print('mean',np.mean(lidar_rect[mask, 2]))

    factor = 0.5
    img = cv2.resize(rgb, (0, 0), fx=factor, fy=factor)
    lidarOnImage[:, 0] *= factor
    lidarOnImage[:, 1] *= factor

    # out = np.zeros((img.shape[0], img.shape[1]))
    # out[np.int32(lidarOnImage.T[1]), np.int32(lidarOnImage.T[0])] = lidarOnImage.T[2]

    grid = 5
    ex = 80
    out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], grid, ex)
    out = cv2.resize(out, (0, 0), fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_LINEAR)
    out = cv2.blur(out, (15, 15))
    print(out[300, 300])
    plt.figure(figsize=(20, 40))
    print(np.mean(out))
    plt.imsave(os.path.join(result_dir, "depth_map_%06d.png" % cur_id), out)
    cv2.imwrite(os.path.join(result_dir, "depth_map_%06d.png" % cur_id), out)
    heatmap = cv2.imread(os.path.join(result_dir, "depth_map_%06d.png" % cur_id))

    depth = cv2.imread(os.path.join(result_dir, "depth_map_%06d.png" % cur_id), cv2.IMREAD_GRAYSCALE)

    print(np.mean(depth[300, 300]))
    print(np.mean(depth))

    np.save(os.path.join(result_dir, "test.npy"), out)
    depth = np.load(os.path.join(result_dir, "test.npy"))


    zs, us, vs = sample_depth_values_and_coordinates(depth)

    fx = 2351.85909
    fy = 2353.56734
    cx = 1017.24032
    cy = 788.70774

    x_cam = zs * (us - cx) / fx
    y_cam = zs * (vs - cy) / fy

    hom = np.ones_like(x_cam.reshape(1000, 1))
    # 3d points in camera frame from depth maps
    points_in_cam_hom = np.stack((x_cam.reshape(1000, 1), y_cam.reshape(1000, 1), zs.reshape(1000, 1), hom), axis=1)

    print(np.mean(zs), np.mean(lidar_rect[:,2]))

    #concat = cv2.hconcat([rgb, heatmap])
    #cv2.imwrite(os.path.join(concat_dir, "depth_map_%06d.png" % cur_id), concat)
    #cv2.imwrite(os.path.join("depth_map_grid_blur_"+str(grid)+ "_factor_"+str(factor)+"_ex_"+str(ex)+"_%06d.png" % cur_id), heatmap)"""
