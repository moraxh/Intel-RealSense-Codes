import warnings
import os
import threading
import winsound
import cv2
import pyrealsense2 as rs
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

from PoseDetector import PoseDetector

FRAME_RATE = 30
CAPTURE_SECONDS = 2
RESOLUTION = (640, 480)
MAX_WIDTH_WINDOW = 1920
OUTPUT_PATH=os.path.join(os.path.dirname(__file__), "captured_data")
CAPTURE_RGB = True
CAPTURE_SKELETON = True

imgs2take = FRAME_RATE * CAPTURE_SECONDS
capture = False

context = rs.context()
devices = context.devices

if len(devices) == 0:
    print("No RealSense device connected. Exiting.")
    exit()

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
  config.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FRAME_RATE)

  # Start pipeline
  pipeline.start(config)
  pipelines.append(pipeline)

  print(f"Device {i} connected, serial number: {serial}")

def save_capture(xyz_path, rgb_path, skeleton_path, lmList, pose, skeleton, imgsCount):
    # Save XYZ data
    np.savetxt(
        fname=f"{xyz_path}/capture_{imgsCount + 1}.csv",
        X=lmList,
        delimiter=","
    )

    # Save the RGB image
    if CAPTURE_RGB:
        cv2.imwrite(
            f"{rgb_path}/capture_{imgsCount + 1}.png",
            pose
        )

    # Save the skeleton
    if CAPTURE_SKELETON:
        cv2.imwrite(
            f"{skeleton_path}/capture_{imgsCount + 1}.png",
            skeleton
        )

def showCamera(i, pipeline, singleCamera):
  global globalCapture
  globalCapture = False
  capture = False
  imgsCount = 0
  detector = PoseDetector()
  object_to_track = range(0,33)

  while True:
    frames = pipeline.wait_for_frames()
    frame_depth = frames.get_depth_frame()
    frame_color = frames.get_color_frame()

    if not frame_color:
      continue

    image_color = np.asanyarray(frame_color.get_data())

    pose, skeleton = detector.findPose(image_color)
    lmList = detector.getPosition(pose)

    if len(lmList) != 0:
      for objet in object_to_track:
          _, x, y = lmList[objet]
          if (x < 0 or x >= RESOLUTION[0]):
            lmList[objet] = [lmList[objet][0], 0, 0, 0]
            continue
          if (y < 0 or y >= RESOLUTION[1]):
            lmList[objet] = [lmList[objet][0], 0, 0, 0]
            continue
          z = frame_depth.get_distance(x, y)
          if (z <= 0): 
            lmList[objet] = [lmList[objet][0], 0, 0, 0]
            continue

          lmList[objet].append(z)

    imgs[i] = pose
    skeletons[i] = skeleton

    if capture or globalCapture:
      # Create output path if not exists
      if not 'path' in locals():
        now  = datetime.now() # Get current timestamp
        path = f"{OUTPUT_PATH}/{now.strftime('%Y_%m_%d_%H_%M_%S')}" # Get timestamp formatted

      capture = True

      if singleCamera:
        xyz_path = f"{path}/xyz"
        rgb_path = f"{path}/rgb"
        skeleton_path = f"{path}/skeleton"
      else:
        xyz_path = f"{path}/camera_{i + 1}/xyz"
        rgb_path = f"{path}/camera_{i + 1}/rgb"
        skeleton_path = f"{path}/camera_{i + 1}/skeleton"

      if not os.path.exists(xyz_path):
        os.makedirs(xyz_path)
      if not os.path.exists(rgb_path) and CAPTURE_RGB:
        os.makedirs(rgb_path)
      if not os.path.exists(skeleton_path) and CAPTURE_SKELETON:
        os.makedirs(skeleton_path)

      cv2.putText(pose, f'Captures: {imgsCount}/{imgs2take}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                  cv2.LINE_AA)

      # Save the capture
      threading.Thread(target=save_capture, args=(xyz_path, rgb_path, skeleton_path, lmList, pose, skeleton, imgsCount)).start()

      imgsCount += 1

      if (imgsCount >= imgs2take):
        playSound()
        imgsCount = 0
        capture = False
        globalCapture = False
        del path

    updateWindow()

    # Start the capture
    if cv2.waitKey(1) == 32:
      globalCapture = True
      capture = True

def playSound():
  winsound.MessageBeep(winsound.MB_OK)

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
    thread = threading.Thread(target=showCamera, args=(i, pipeline, len(pipelines) == 1))
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