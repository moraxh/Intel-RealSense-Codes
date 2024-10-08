import numpy as np
import open3d as o3d

filepath = "captured_data/2024_08_19_14_48_45/capture_1.csv"

# Get data from file
xyz = np.loadtxt(filepath, delimiter=",", dtype=float)

# ----------------------------
# -------- FILTER ------------
# ----------------------------

# Get norm-2 for every point
xyz_norm = np.linalg.norm(xyz, axis=1)

# Print info 
print(f"min_distance: {round(min(xyz_norm) * 100, 4)}cm, max_distance: {round(max(xyz_norm) * 100, 4)}cm")

# Set min and max distance
min_dist = 0.00
max_dist = 10.00 # Capture 4
 
# Filter points based on distance
mask = (xyz_norm > min_dist) & (xyz_norm < max_dist)

xyz = xyz[mask]

# Create an point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Display the point cloud object
o3d.visualization.draw_geometries([pcd])