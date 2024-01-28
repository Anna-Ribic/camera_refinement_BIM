import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import open3d as o3d

est_poses = np.load('../con/data/test2/checkpoints/est_poses.npy')

for trans in est_poses:
    point = np.zeros((4, 1))
    point[-1] = 1
    cam = trans @ point
    cam = cam[:-1].squeeze()
    with open('est_traj.obj', 'a') as ob:
        ob.write('v ' + str(cam[0].item()) + ' ' + str(cam[1].item()) + ' ' + str(cam[2].item()))
        ob.write('\n')
