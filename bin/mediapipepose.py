print("Importing libraries...")

import os
import sys

sys.path.append(os.getcwd())    #embedable python doesnt find local modules without this line

import time
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt

from helpers import  CameraStream, Skeleton, LandmarkPose, _3dPoint, shutdown, get_rot_mediapipe, get_rot_hands, draw_pose, keypoints_to_original, normalize_screen_coordinates, get_rot
from scipy.spatial.transform import Rotation as R
from backends import DummyBackend, SteamVRBackend, VRChatOSCBackend, Backend
import webui

import inference_gui
import pyplot_gui
import parameters

import tkinter as tk

import mediapipe as mp
from mediapipe.python.solutions import drawing_utils, pose


def main():
    mp_drawing = drawing_utils
    mp_pose = pose

    use_steamvr = True

    print("INFO: Reading parameters...")

    params = parameters.Parameters()
    
    if params.webui:
        webui_thread = threading.Thread(target=webui.start_webui, args=(params,), daemon=True)
        webui_thread.start()
    else:
        print("INFO: WebUI disabled in parameters")

    backends = { 0: DummyBackend, 1: SteamVRBackend, 2: VRChatOSCBackend }
    backend:Backend = backends[params.backend]()
    backend.connect(params)

    if params.exit_ready:
        sys.exit("INFO: Exiting... You can close the window after 10 seconds.")

    print("INFO: Opening camera...")

    camera_thread = CameraStream(params)

    #making gui
    # make_inference_gui(params)
    gui_thread = threading.Thread(target=inference_gui.make_inference_gui, args=(params,), daemon=True)
    gui_thread.start()

    Landmark = LandmarkPose()
    VR_skeleton = Skeleton()

    # # Pyplot
    # pyplot_thread = threading.Thread(target=pyplot_gui.make_pyplot_gui, args =(params, VR_skeleton), daemon = True )
    # pyplot_thread.start()

    print("INFO: Starting pose detector...")

    #create our detector. These are default parameters as used in the tutorial. 
    pose_detector = mp_pose.Pose(model_complexity=params.model, 
                                 min_detection_confidence=0.5, 
                                 min_tracking_confidence=params.min_tracking_confidence, 
                                 smooth_landmarks=params.smooth_landmarks, 
                                 static_image_mode=params.static_image)
    

    cv2.namedWindow("out")

    #Main program loop:
    prev_smoothing = params.smoothing
    prev_add_smoothing = params.additional_smoothing


    while not params.exit_ready:
        if prev_smoothing != params.smoothing or prev_add_smoothing != params.additional_smoothing:
            print(f"INFO: Changed smoothing value from {prev_smoothing} to {params.smoothing}")
            print(f"INFO: Changed additional smoothing from {prev_add_smoothing} to {params.additional_smoothing}")

            prev_smoothing = params.smoothing
            prev_add_smoothing = params.additional_smoothing

            backend.onparamchanged(params)

        #wait untill camera thread captures another image
        if not camera_thread.image_ready:     
            time.sleep(0.001)
            # print(". ", end='')
            continue

        #some may say I need a mutex here. I say this works fine.
        img = camera_thread.image_from_thread.copy() 
        camera_thread.image_ready = False
        
        #if set, rotate the image
        if params.rotate_image is not None:       
            img = cv2.rotate(img, params.rotate_image)
            
        if params.mirror:
            img = cv2.flip(img,1)

        #if set, ensure image does not exceed the given size.
        if max(img.shape) > params.maximgsize:        
            ratio = max(img.shape)/params.maximgsize
            img = cv2.resize(img, (int(img.shape[1]/ratio),int(img.shape[0]/ratio)))
        
        if params.paused:
            cv2.imshow("out", img)
            cv2.waitKey(1)
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.    copied from the tutorial
        img.flags.writeable = False
        landmark_pose = pose_detector.process(img)
        img.flags.writeable = True


        if landmark_pose.pose_world_landmarks:        #if any pose was detected
            
            Landmark.update_landmarks(landmark_pose.pose_world_landmarks.landmark)
            VR_skeleton.update_skeleton(Landmark)   #convert keypoints to a format we use
            
            #do we need this with osc as well?

            VR_skeleton_og = VR_skeleton.copy()
            params.VR_skeleton_og = VR_skeleton_og

            # if PlotCounter <= 60:
            #     PlotPoints = False
            #     PlotCounter += 1
            # else:
            #     PlotPoints = True
            #     PlotCounter = 0

            for key, value in VR_skeleton.skeleton_points.items():        #apply the rotations from the sliders
                VR_skeleton.skeleton_points[key] = params.global_rot_z.apply(VR_skeleton.skeleton_points[key])
                VR_skeleton.skeleton_points[key] = params.global_rot_x.apply(VR_skeleton.skeleton_points[key])
                VR_skeleton.skeleton_points[key] = params.global_rot_y.apply(VR_skeleton.skeleton_points[key])
            
            if not params.feet_rotation:
                rots = get_rot(VR_skeleton)          #get rotation data of feet and hips from the position-only skeleton data
            else:
                rots = get_rot_mediapipe(VR_skeleton)
                
            if params.use_hands:
                hand_rots = get_rot_hands(VR_skeleton)
            else:
                hand_rots = None
            
            if not backend.updatepose(params, VR_skeleton, rots, hand_rots):
                continue
        
        
        #print(f"Inference time: {time.time()-t0}\nSmoothing value: {smoothing}\n")        #print how long it took to detect and calculate everything
        inference_time = time.time() - t0
        
        if params.log_frametime:
            print(f"INFO: Inference time: {inference_time}")
        
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)       #convert back to bgr and draw the pose
        mp_drawing.draw_landmarks(
            img, landmark_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        img = cv2.putText(img, f"{inference_time:1.3f}, FPS:{int(1/inference_time)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("out", img)           #show image, exit program if we press esc
        if cv2.waitKey(1) == 27:
            backend.disconnect()
            shutdown(params)

    # Capture frame-by-frame
    if params.exit_ready:
        shutdown(params)

if __name__ == "__main__":
    main()

