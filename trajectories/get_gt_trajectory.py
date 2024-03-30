import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import open3d as o3d
import decimal

#Get translation of trajectory to ply file for visualization reasons
with open("offset_trajectory_0.03_2.txt", 'r') as file:
    for line in file:
        # Split the line into timestamp, translation, and rotation parts
        parts = line.split()
        timestamp = int(decimal.Decimal(parts[0])*1000000)
        translation = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        rotation = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])

        with open('offset_trajectory_0.03_2.obj', 'a') as ob:
            ob.write('v ' + str(translation[0].item()) + ' ' + str(translation[1].item()) + ' ' + str(translation[2].item()))
            ob.write('\n')
