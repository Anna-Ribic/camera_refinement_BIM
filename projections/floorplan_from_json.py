import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import json

def write_ply_file(segments_array, filename):
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(segments_array) * 20))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element edge {}\n".format(len(segments_array)))
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        # Write vertex data
        for segment in segments_array:
            start_point = segment[:2]
            end_point = segment[2:]
            for t in np.linspace(0, 1, 21)[:-1]:  # Divide segment into 20 equidistant points
                x = start_point[0] + t * (end_point[0] - start_point[0])
                y = start_point[1] + t * (end_point[1] - start_point[1])
                z = 15  # Z coordinate is assumed to be 15
                f.write("{:.6f} {:.6f} {:.6f}\n".format(x, y, z))

        # Write edge data
        vertex_idx = np.arange(len(segments_array) * 20).reshape(-1, 20)
        for i, segment_vertices in enumerate(vertex_idx):
            for j in range(20 - 1):
                f.write("{} {}\n".format(segment_vertices[j], segment_vertices[j + 1]))
            f.write("{} {}\n".format(segment_vertices[0], segment_vertices[-1]))

bim_path = '../output/bim.ply'
pcd = o3d.io.read_point_cloud(bim_path)

point_cloud_data = np.asarray(pcd.points)

x_origin = np.min(point_cloud_data[:, 0])
y_origin = np.min(point_cloud_data[:, 1])

print('origin', x_origin, y_origin)

with open('../output/floorplan.json', 'r') as f:
    data = json.load(f)

# Initialize an empty list to store segment data
segments = []

# Iterate over each segment in the JSON data
for segment in data:
    start_point = [segment['start_point']['x'], segment['start_point']['y']]
    end_point = [segment['end_point']['x'], segment['end_point']['y']]
    segments.append(start_point + end_point)  # Combine start and end points

# Convert the list of segments to a numpy array
segments_array = np.array(segments) * 0.05
print(segments_array)
segments_array[:, [0,2]] += y_origin - 1.02
segments_array[:, [1,3]] += x_origin - 3.62


np.save('../output/floorplan_inverted.npy', segments_array)
np.savetxt('../output/floorplan_inverted.txt', segments_array)

write_ply_file(segments_array, '../output/floorplan_inverted.ply')



