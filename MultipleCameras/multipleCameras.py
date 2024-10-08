import warnings
import os
import threading
import time
import cv2
import pyrealsense2 as rs
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

from PoseDetector import PoseDetector

FRAME_RATE = 30
CAPTURE_SECONDS = 5
RESOLUTION = (640, 480)
MAX_WIDTH_WINDOW = 1920
OUTPUT_PATH="captured_data"

imgs2take = FRAME_RATE * CAPTURE_SECONDS
path = None

context = rs.context()
devices = context.devices

if len(devices) == 0:
    print("No RealSense device connected. Exiting.")
    exit()

if (not os.path.exists(path)):
  os.makedirs(path)

# Pipeline
pipelines = []
for i, device in enumerate(devices):
  pipeline = rs.pipeline()
  config =  rs.config()

  # Use the specific device using the serial
  serial = device.get_info(rs.camera_info.serial_number)
  config.enable_device(serial)

  # Enable streams
  config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FRAME_RATE)

  # Start pipeline
  pipeline.start(config)
  pipelines.append(pipeline)

  print(f"Device {i} connected, serial number: {serial}")

def showCamera(i, pipeline):
  capture = False
  imgsCount = 0
  detector = PoseDetector()
  object_to_track = range(0,33)

  while True:
    frames = pipeline.wait_for_frames()
    frame_color = frames.get_color_frame()

    if not frame_color:
      continue

    image_color = np.asanyarray(frame_color.get_data())

    pose, skeleton = detector.findPose(image_color)
    lmList = detector.getPosition(pose)

    if len(lmList) != 0:
        for objet in object_to_track:
            _, x, y = lmList[objet]
            if (x < 0 or x > RESOLUTION[0]):
                del objet
                continue
            if (y < 0 or y > RESOLUTION[1]):
                del objet
                continue

    imgs[i] = pose
    skeletons[i] = skeleton

    if capture:
      cv2.putText(pose, f'Captures: {imgsCount}/{imgs2take}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                  cv2.LINE_AA)

      # Save
      # Save the points
      np.savetxt(
          fname     = f"{xyz_path}/capture_{imgsCount}.csv",
          X         = lmList,
          delimiter = ","
      )

      # Save the image
      cv2.imwrite(
          f"{rgb_path}/capture_{i + 1}.png",
          pose
      )

      # Save the skeleton
      cv2.imwrite(
          f"{skeleton_path}/capture_{i + 1}.png",
          skeleton
      )
      
      imgsCount += 1

      if (imgsCount >= imgs2take):
        imgsCount = 0
        capture = False

    updateWindow()

    # Start the capture
    if cv2.waitKey(1) == 32:
      # Create output path if not exists
      if path == None:
        now  = datetime.now() # Get current timestamp
        path = f"{OUTPUT_PATH}/{now.strftime('%Y_%m_%d_%H_%M_%S')}" # Get timestamp formatted

      capture = True
      xyz_path = f"{path}/camera_{i}/xyz"
      rgb_path = f"{path}/camera_{i}/rgb"
      skeleton_path = f"{path}/camera_{i}/skeleton"

      if not os.path.exists(xyz_path):
        os.makedirs(xyz_path)
      if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
      if not os.path.exists(skeleton_path):
        os.makedirs(skeleton_path)

def updateWindow():
  arrange_color = np.concatenate(imgs, axis=1)
  arrange_skeletons = np.concatenate(skeletons, axis=1)

  arrange_imgs = np.concatenate((arrange_color, arrange_skeletons), axis=0)

  if (arrange_imgs.shape[1] > MAX_WIDTH_WINDOW):
      arrange_imgs = cv2.resize(arrange_imgs, (MAX_WIDTH_WINDOW, int(arrange_imgs.shape[0] / arrange_imgs.shape[1] * MAX_WIDTH_WINDOW)))
  cv2.imshow("RealSense", arrange_imgs)

threads = []

imgs = [np.zeros((RESOLUTION[1], RESOLUTION[0], 3), np.uint8) for i in range(len(devices))]
skeletons = [np.zeros((RESOLUTION[1], RESOLUTION[0], 3), np.uint8) for i in range(len(devices))]

try:
  for i, pipeline in enumerate(pipelines):
    thread = threading.Thread(target=showCamera, args=(i, pipeline))
    threads.append(thread)
    thread.start()
finally:
  # Stop all threads
  for thread in threads:
     thread.join()
  # Stop all pipelines
  for pipeline in pipelines:
    pipeline.stop()
  cv2.destroyAllWindows()