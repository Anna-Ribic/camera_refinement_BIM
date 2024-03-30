import open3d as o3d
import numpy as np
import map_metrics
from scipy.stats import entropy
from scipy.spatial import cKDTree


def downsample_point_cloud(point_cloud, voxel_size):
    return point_cloud.voxel_down_sample(voxel_size=voxel_size)


def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)


def mean_map_entropy(pcs, Ts, radius=0.1, num_neighbors=10):
    """
    Calculate the mean map entropy of a set of point clouds.

    Args:
        pcs (list): List of point clouds (open3d.PointCloud).
        Ts (list): List of corresponding poses in the trajectory (4x4 transformation matrices).
        radius (float): Radius for finding neighbors.
        num_neighbors (int): Number of neighbors to consider for entropy calculation.

    Returns:
        float: Mean map entropy.
    """
    # Initialize an empty array to store entropy values for each point
    point_entropies = []

    # Iterate through each point cloud and its corresponding pose
    for pc, T in zip(pcs, Ts):
        # Transform the point cloud according to the pose
        pc.transform(T)

        # Convert the point cloud to a numpy array
        points = np.asarray(pc.points)

        # Build a KD-tree for fast nearest neighbor search
        kdtree = cKDTree(points)

        # Calculate entropy for each point
        entropies = []
        for point in points:
            # Find nearest neighbors within the specified radius
            _, neighbor_indices = kdtree.query(point, k=num_neighbors, distance_upper_bound=radius)

            # Filter out invalid neighbor indices
            neighbor_indices = neighbor_indices[neighbor_indices < len(points)]

            # Get the coordinates of the neighbors
            neighbors = points[neighbor_indices]

            # Calculate probability distribution
            _, counts = np.unique(neighbor_indices, return_counts=True)
            prob_dist = counts / np.sum(counts)

            # Calculate entropy
            entropies.append(entropy(prob_dist))

        # Append entropy values for points in this point cloud
        point_entropies.extend(entropies)

    # Calculate the mean entropy over all points
    mean_entropy = np.mean(point_entropies)

    return mean_entropy

#Modify
path = 'facap/source_pcd.ply'

print(path)
pc = load_point_cloud(path)
# ground_truth_point_cloud = load_point_cloud("lowres_p_gfw/gt_pcd.ply")

print(len(pc.points))

# pcd = downsample_point_cloud(pc, voxel_size=0.1)
# print(len(pcd.points))

mme = map_metrics.mme([np.asarray(pc.points).astype(np.float64).T], [np.eye(4).astype(np.float64)])
print('mme', mme)

mpv = map_metrics.mpv([np.asarray(pc.points).astype(np.float64).T], [np.eye(4).astype(np.float64)])
print('mpv', mpv)

mom = map_metrics.mom([np.asarray(pc.points).astype(np.float64).T], [np.eye(4).astype(np.float64)])

print('mom', mom)