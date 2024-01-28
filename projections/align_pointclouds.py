import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    """Load a point cloud from a file."""
    cloud = o3d.io.read_point_cloud(file_path)
    return cloud

def align_point_clouds(source_cloud, target_cloud):
    """Align two point clouds using ICP."""
    icp_transformation = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, max_correspondence_distance=25,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
    )

    aligned_source_cloud = source_cloud.transform(icp_transformation.transformation)
    return aligned_source_cloud

def save_point_cloud(file_path, point_cloud):
    """Save a point cloud to a file."""
    o3d.io.write_point_cloud(file_path, point_cloud)

# Load point clouds
ply_path = "../output/bim.ply"
obj_path = "../output/pcfull.ply"

source_cloud = np.asarray(load_point_cloud(ply_path).points).astype(np.float32)
target_cloud = np.asarray(load_point_cloud(obj_path).points).astype(np.float32)

source_center = np.mean(source_cloud, axis=0)
print(source_center)
target_center = np.mean(target_cloud, axis=0)
print(target_center)
diff = target_center - source_center
print(diff)
source_cloud = source_cloud + diff
print(np.mean(source_cloud))

###!!!!MAke notebook from this -> better
source_cloud_o3 = o3d.geometry.PointCloud()
source_cloud_o3.points = o3d.utility.Vector3dVector(source_cloud)

target_cloud_o3 = o3d.geometry.PointCloud()
target_cloud_o3.points = o3d.utility.Vector3dVector(target_cloud)

# Align point clouds
aligned_source_cloud = align_point_clouds(source_cloud_o3, target_cloud_o3)

print(np.mean(aligned_source_cloud.points, axis=0))
print(np.mean(target_cloud_o3.points, axis=0))

# Save the aligned point cloud
aligned_ply_path = "../output/aligned.ply"
save_point_cloud(aligned_ply_path, aligned_source_cloud)