import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import os
import cv2

def compute_pose_rmse(translations_gt, rotations_gt, translations_est, rotations_est):
    # Compute squared errors for translations
    translation_errors = np.linalg.norm(translations_gt - translations_est, axis=1) ** 2
    ate_rmse = np.sqrt(np.mean(translation_errors))

    # Compute squared errors for rotations
    degree_errors = []
    yaw_erros = []
    pitch_erros = []
    roll_erros = []
    for rotvec_gt, rotvec_est in zip(rotations_gt, rotations_est):
        rotation_gt = R.from_rotvec(rotvec_gt)
        rotation_est = R.from_rotvec(rotvec_est)

        gt_euler = rotation_gt.as_euler('xyz', degrees=True)

        est_euler = rotation_est.as_euler('xyz', degrees=True)

        yaw_difference = est_euler[2] - gt_euler[2]
        pitch_difference = est_euler[0] - gt_euler[0]
        roll_difference = est_euler[1] - gt_euler[1]

        yaw_difference = min(abs(yaw_difference), abs(360-yaw_difference), abs(360+yaw_difference))
        pitch_difference = min(abs(pitch_difference), abs(360-pitch_difference), abs(360+pitch_difference))
        roll_difference = min(abs(roll_difference), abs(360-roll_difference), abs(360+roll_difference))

        print("Difference in yaw (degrees):", yaw_difference)
        yaw_erros.append(yaw_difference)
        print("Difference in pitch (degrees):", pitch_difference)
        pitch_erros.append(pitch_difference)
        print("Difference in roll (degrees):", roll_difference)
        roll_erros.append(roll_difference)
        total_difference = yaw_difference + pitch_difference + roll_difference
        degree_errors.append(total_difference)
        print("Total Difference:" , total_difference)


    rot_rmse = np.sqrt(np.mean(np.square(degree_errors)))
    yaw_rmse = np.sqrt(np.mean(np.square(yaw_erros)))
    pitch_rmse = np.sqrt(np.mean(np.square(pitch_erros)))
    roll_rmse = np.sqrt(np.mean(np.square(roll_erros)))

    return ate_rmse, rot_rmse/10, yaw_rmse/10, pitch_rmse/10, roll_rmse/10

path = '../scan_lowres_0_5/results_facap'

print(path)

cam_gt = torch.load(os.path.join(path, 'gt_cameras.pth'),map_location=torch.device('cpu') )
cam_est = torch.load(os.path.join(path, 'source_cameras.pth'), map_location=torch.device('cpu'))

tr, rr, yr, pr, rolr = compute_pose_rmse(cam_gt['translations'], cam_gt['rotvecs'],cam_est['translations'], cam_est['rotvecs'])

print('trmse', tr)
print('rrmse', rr)
print('yawmse', yr)
print('prmse', pr)
print('rollrmse', rolr)