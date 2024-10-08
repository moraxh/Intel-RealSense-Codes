# Libraries
import os
import cv2
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from datetime import datetime

def filterCloudPoint(xyz):
    # Get norm-2 for every point
    xyz_norm = np.linalg.norm(xyz, axis=1)

    # Set min and max distance
    min_dist = 0.00
    max_dist = 0.40 
    
    # Filter points based on distance
    mask = (xyz_norm > min_dist) & (xyz_norm < max_dist)

    return xyz[mask]

def saveCallback(vis, i):
    global saveCloudPoint
    saveCloudPoint = True

    # Capture image
    vis.capture_screen_image(f"{cloudPointsImg_path}/capture_{i + 1}.png")

    # Destroy window
    vis.destroy_window()

OUTPUT_PATH    = "captured_data";
CAPTURE_FRAMES = 200

saveCloudPoint = False

# Create pipelines
pipeline = rs.pipeline()
config   = rs.config()

# Config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start transmission
pipeline.start(config)

# Create timestamp dir
now                 = datetime.now() # Get current timestamp
path                = f"{OUTPUT_PATH}/{now.strftime('%Y_%m_%d_%H_%M_%S')}" # Get timestamp formatted
cloudPoints_path    = f"{path}/cloud_points"
rgb_path            = f"{path}/rgb"
depth_path          = f"{path}/depth"
cloudPointsImg_path = f"{path}/cloud_points_image"

# Create root folder
os.makedirs(path, exist_ok=True)
# Create cloud points path
os.makedirs(cloudPoints_path, exist_ok=True)
# Create rgb folder
os.makedirs(rgb_path, exist_ok=True)
# Create depth color
os.makedirs(depth_path, exist_ok=True)
# Create cloud points img path
os.makedirs(cloudPointsImg_path, exist_ok=True)

i = 0

try:
    while i < CAPTURE_FRAMES:
        saveCloudPoint = False

        ##############################
        ######## PRE STREAM ##########
        ##############################
        while True:
            # Get frames
            frames      = pipeline.wait_for_frames()
            frame_color = frames.get_color_frame()
            frame_depth = frames.get_depth_frame()

            # Frames to arrays
            image_color = np.asanyarray(frame_color.get_data())
            image_depth = np.asanyarray(frame_depth.get_data()) 

            colormap = cv2.applyColorMap(cv2.convertScaleAbs(image_depth, alpha=0.01), cv2.COLORMAP_JET)

            colormap_depth_dim = colormap.shape
            colormap_color_dim = image_color.shape

            # Resize to match sizes
            if colormap_depth_dim != colormap_color_dim:
                resized_color_image = cv2.resize(image_color, dsize=(colormap_depth_dim[1], colormap_depth_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, colormap))
            else:
                images = np.hstack((image_color, colormap))


            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense',images)

            # Exit with esc or enter
            key = cv2.waitKey(1)
            if key & key == 13 or key == 27 or key == 32: # Esc, enter or space
                cv2.destroyAllWindows()
                break
            
        #########################
        ####### CAPTURE #########
        ########################!

        # Get frames
        frame       = pipeline.wait_for_frames()
        frame_depth = frame.get_depth_frame()
        frame_color = frame.get_color_frame()
        
        # Skip frame if depth or color frame doesn't exists
        if not frame_depth or not frame_color:
            continue

        # To numpy array
        image_depth = np.asanyarray(frame_depth.get_data())
        image_color = np.asanyarray(frame_color.get_data())

        colormap = cv2.applyColorMap(cv2.convertScaleAbs(image_depth, alpha=0.03), cv2.COLORMAP_JET)

        # Create point cloud
        pc     = rs.pointcloud()
        points = pc.calculate(frame_depth)
        vtx    = np.asanyarray(points.get_vertices())
        xyz    = np.asanyarray(vtx).view(np.float32).reshape(-1, 3)

        ###############################
        ######### PRE PROCESS #########
        ###############################

        xyz = filterCloudPoint(xyz)

        #################################
        ######### VISUALIZATION #########
        #################################

        # Invert axis y and z
        xyz[:, 1] *= -1
        xyz[:, 2] *= -1
        
        # Create the visualizer
        pcd        = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Create the visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name='Cloud Point', width=1920, height=1080, left=0, top=0)

        # Register key callbacks
        vis.register_key_callback(32, lambda vis: saveCallback(vis, i))

        # Add the cloud points
        vis.add_geometry(pcd)
        vis.update_renderer()

        # Start the visualization
        vis.run()

        # Destroy window
        vis.destroy_window()

        ########################
        ####### SAVING #########
        ########################
       
        if saveCloudPoint:
            # Save the point cloud
            np.savetxt(
                fname     = f"{cloudPoints_path}/capture_{i + 1}.csv",
                X         = xyz,
                delimiter = ","
            )

            # Save the image in rgb
            cv2.imwrite(
                f"{rgb_path}/capture_{i + 1}.png",
                image_color
            )

            # Save the image in colormap
            cv2.imwrite(
                f"{depth_path}/capture_depth_{i + 1}.png",
                colormap
            )

            print(f"{i + 1}/{CAPTURE_FRAMES}")

            i = i + 1
        else:
            print(f"Rejected frame no.{i + 1}")

finally:
    # Stop the transmission
    pipeline.stop()
    