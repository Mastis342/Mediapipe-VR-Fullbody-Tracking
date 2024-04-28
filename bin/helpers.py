import time

import socket
import cv2

import numpy as np
from sys import platform
from scipy.spatial.transform import Rotation as R
import cv2
import threading
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from parameters import Parameters

def draw_pose(frame,pose,size):
    pose = pose*size
    for sk in EDGES:
        cv2.line(frame,(int(pose[sk[0],1]),int(pose[sk[0],0])),(int(pose[sk[1],1]),int(pose[sk[1],0])),(0,255,0),3)

class _3dPoint():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other:'_3dPoint'):
        Result = _3dPoint(self.x + other.x,
                          self.y + other.y,
                          self.z + other.z) 
        return Result

    def __truediv__(self, divisor:float):
        Result = _3dPoint(self.x/divisor,
                          self.y/divisor,
                          self.z/divisor)
        return Result
    
    def __mul__(self, factor:float):
        Result = _3dPoint(self.x * factor,
                          self.y * factor,
                          self.z * factor)
        return Result

    def __mul__(self, other:'_3dPoint'):
        Result = _3dPoint(self.x * other.x +
                          self.y * other.y +
                          self.z * other.z)
        return Result

    
    def generate_np_vector_from_3d_point(self):
        '''
        Generate an np vector representation that is usable by SteamVR
        x and y direction are flipped for SteamVR

        Input:
            _3dPoint
        Output:
            3x1 np vector
        '''
        return np.array([-self.x, -self.y, self.z])

class FootRotation():
    '''
    Class to save foot rotation
    '''
    def __init__(self, RightFoot, LeftFoot, Hip) -> None:

        self.RightFoot = RightFoot
        self.LeftFoot  = LeftFoot
        self.Hip       = Hip

class LandmarkPose():
    '''
    Class to save a landmark pose
    '''
    def __init__(self) -> None:
        #33 pose landmarks as in https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
        #convert landmarks returned by mediapipe to skeleton that I use.
        #lms = results.pose_world_landmarks.landmark
        self.lms = None

        self.LMNose             = None
        self.LMRightEyeInner    = None
        self.LMRightEye         = None
        self.LMRightEyeOuter    = None
        self.LMLeftEyeInner     = None
        self.LMLeftEye          = None
        self.LMLeftEyeOuter     = None
        self.LMRightEar         = None
        self.LMLeftEar          = None
        self.LMMouthRight       = None
        self.LMMouthLeft        = None
        self.LMRightShoulder    = None
        self.LMLeftShoulder     = None
        self.LMRightElbow       = None
        self.LMLeftElbow        = None
        self.LMRightWrist       = None
        self.LMLeftWrist        = None
        self.LMRightPinky       = None
        self.LMLeftPinky        = None
        self.LMRightIndex       = None
        self.LMLeftIndex        = None
        self.LMRightThumb       = None
        self.LMLeftThumb        = None
        self.LMRightHip         = None
        self.LMLeftHip          = None
        self.LMRightKnee        = None
        self.LMLeftKnee         = None
        self.LMRightAnkle       = None
        self.LMLeftAnkle        = None
        self.LMRightHeel        = None
        self.LMLeftHeel         = None
        self.LMRightFootIndex   = None
        self.LMLeftFootIndex    = None

    def update_landmarks(self, lms):
        self.lms = lms

        self.LMNose             = self.generate_3d_point_from_landmark(lms[0])
        self.LMRightEyeInner    = self.generate_3d_point_from_landmark(lms[1])
        self.LMRightEye         = self.generate_3d_point_from_landmark(lms[2])
        self.LMRightEyeOuter    = self.generate_3d_point_from_landmark(lms[3])
        self.LMLeftEyeInner     = self.generate_3d_point_from_landmark(lms[4])
        self.LMLeftEye          = self.generate_3d_point_from_landmark(lms[5])
        self.LMLeftEyeOuter     = self.generate_3d_point_from_landmark(lms[6])
        self.LMRightEar         = self.generate_3d_point_from_landmark(lms[7])
        self.LMLeftEar          = self.generate_3d_point_from_landmark(lms[8])
        self.LMMouthRight       = self.generate_3d_point_from_landmark(lms[9])
        self.LMMouthLeft        = self.generate_3d_point_from_landmark(lms[10])
        self.LMRightShoulder    = self.generate_3d_point_from_landmark(lms[11])
        self.LMLeftShoulder     = self.generate_3d_point_from_landmark(lms[12])
        self.LMRightElbow       = self.generate_3d_point_from_landmark(lms[13])
        self.LMLeftElbow        = self.generate_3d_point_from_landmark(lms[14])
        self.LMRightWrist       = self.generate_3d_point_from_landmark(lms[15])
        self.LMLeftWrist        = self.generate_3d_point_from_landmark(lms[16])
        self.LMRightPinky       = self.generate_3d_point_from_landmark(lms[17])
        self.LMLeftPinky        = self.generate_3d_point_from_landmark(lms[18])
        self.LMRightIndex       = self.generate_3d_point_from_landmark(lms[19])
        self.LMLeftIndex        = self.generate_3d_point_from_landmark(lms[20])
        self.LMRightThumb       = self.generate_3d_point_from_landmark(lms[21])
        self.LMLeftThumb        = self.generate_3d_point_from_landmark(lms[22])
        self.LMRightHip         = self.generate_3d_point_from_landmark(lms[23])
        self.LMLeftHip          = self.generate_3d_point_from_landmark(lms[24])
        self.LMRightKnee        = self.generate_3d_point_from_landmark(lms[25])
        self.LMLeftKnee         = self.generate_3d_point_from_landmark(lms[26])
        self.LMRightAnkle       = self.generate_3d_point_from_landmark(lms[27])
        self.LMLeftAnkle        = self.generate_3d_point_from_landmark(lms[28])
        self.LMRightHeel        = self.generate_3d_point_from_landmark(lms[29])
        self.LMLeftHeel         = self.generate_3d_point_from_landmark(lms[30])
        self.LMRightFootIndex   = self.generate_3d_point_from_landmark(lms[31])
        self.LMLeftFootIndex    = self.generate_3d_point_from_landmark(lms[32])

    def generate_3d_point_from_landmark(self, landmark) -> _3dPoint:
        return _3dPoint(landmark.x, landmark.y, landmark.z)

class Skeleton():
    '''
    Returns an 29x3 Matrix with 29 Pose positions saved in x,y,z coordinates
    '''
    def __init__(self):

        self.skeleton_points = {}

        self.Nr_of_skeleton_points = len(self.skeleton_points)

    def __mul__(self, factor) -> 'Skeleton':
        '''
        Multiplies all the vectors belonging to the skeleton by the scale factor
        '''
        NewSkeleton = self.copy()
        
        # for i in dir(self):
        #     if i == 'lms':
        #         continue

        #     if '__' not in i:
        #         # Copy Landmarks and Skeleton
        #         exec(f'NewSkeleton.{i} = self.{i}')
        #         if 'LM' not in i:
        #             # Multiply Skeleton
        #             exec(f'NewSkeleton.{i} = NewSkeleton.{i} * {factor}')
        for key, value in NewSkeleton.skeleton_points.items():
            NewSkeleton.skeleton_points[key] = value * factor

        return NewSkeleton
    
    # def __iter__(self):
    #     return self.Origin

    # def __next__(self):
    #     if self._index < self.Nr_of_skeleton_points:

    def update_skeleton(self, landmark:LandmarkPose):

        self.skeleton_points = {
            'Origin'            : np.array([0,0,0]),
            
            # Right leg
            'RightAnkle'        : landmark.LMRightAnkle.generate_np_vector_from_3d_point(),
            'RightHeel'         : landmark.LMRightHeel.generate_np_vector_from_3d_point(),           # Back
            'RightFootIndex'    : landmark.LMRightFootIndex.generate_np_vector_from_3d_point(),      # Forward
            'RightKnee'         : landmark.LMRightKnee.generate_np_vector_from_3d_point(),           # Up
            'RightHip'          : landmark.LMRightHip.generate_np_vector_from_3d_point(),

            # Left leg
            'LeftAnkle'         : landmark.LMLeftAnkle.generate_np_vector_from_3d_point(),
            'LeftHeel'          : landmark.LMLeftHeel.generate_np_vector_from_3d_point(),            # Back
            'LeftFootIndex'     : landmark.LMLeftFootIndex.generate_np_vector_from_3d_point(),       # Forward
            'LeftKnee'          : landmark.LMLeftKnee.generate_np_vector_from_3d_point(),            # Up
            'LeftHip'           : landmark.LMLeftHip.generate_np_vector_from_3d_point(),

            # Right arm
            'RightWrist'        : landmark.LMRightWrist.generate_np_vector_from_3d_point(),          # Back
            'RightPinky'        : landmark.LMRightPinky.generate_np_vector_from_3d_point(),          # Forward
            'RightIndex'        : landmark.LMRightIndex.generate_np_vector_from_3d_point(),          # Up
            'RightElbow'        : landmark.LMRightElbow.generate_np_vector_from_3d_point(),
            'RightShoulder'     : landmark.LMRightShoulder.generate_np_vector_from_3d_point(),

            # Left arm
            'LeftWrist'         : landmark.LMLeftWrist.generate_np_vector_from_3d_point(),           # Back
            'LeftPinky'         : landmark.LMLeftPinky.generate_np_vector_from_3d_point(),           # Forward
            'LeftIndex'         : landmark.LMLeftIndex.generate_np_vector_from_3d_point(),           # Up
            'LeftElbow'         : landmark.LMLeftElbow.generate_np_vector_from_3d_point(),
            'LeftShoulder'      : landmark.LMLeftShoulder.generate_np_vector_from_3d_point(),

            'Torso'             : ((landmark.LMLeftShoulder + landmark.LMRightShoulder)/2).generate_np_vector_from_3d_point(),
            'Mouth'             : ((landmark.LMMouthLeft + landmark.LMMouthRight)/2).generate_np_vector_from_3d_point(),
            'Nose'              : landmark.LMNose.generate_np_vector_from_3d_point(),

            # 'Test1'             : np.array([1,0,0]),
            # 'Test2'             : np.array([0,1,0]),
            # 'Test3'             : np.array([0,0,1]),

            'HipCenter'         : ((landmark.LMLeftHip + landmark.LMRightHip)/2).generate_np_vector_from_3d_point(),
        }

        self.Nr_of_skeleton_points = len(self.skeleton_points)



    def copy(self):
        Copy = Skeleton()
        for key, value in self.skeleton_points.items():
            Copy.skeleton_points[key] = value
        Copy.Nr_of_skeleton_points = self.Nr_of_skeleton_points

        return Copy

def keypoints_to_original(scale,center,points):
    scores = points[:,2]
    points -= 0.5
    points *= scale
    points[:,0] += center[0]
    points[:,1] += center[1]
    
    points[:,2] = scores
    
    return points

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def get_rot_hands(pose3d):

    hand_r_f = pose3d[26]
    hand_r_b = pose3d[27]
    hand_r_u = pose3d[28]
    
    hand_l_f = pose3d[23]
    hand_l_b = pose3d[24]
    hand_l_u = pose3d[25]
    
    # left hand
    
    x = hand_l_f - hand_l_b
    w = hand_l_u - hand_l_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    l_hand_rot = np.vstack((z, y, -x)).T
    
    # right hand
    
    x = hand_r_f - hand_r_b
    w = hand_r_u - hand_r_b
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    r_hand_rot = np.vstack((z, y, -x)).T

    r_hand_rot = R.from_matrix(r_hand_rot).as_quat()
    l_hand_rot = R.from_matrix(l_hand_rot).as_quat()
    
    return l_hand_rot, r_hand_rot

def get_rot_mediapipe(skeleton:Skeleton) -> FootRotation:
    
    ############################################################# 
    # Hip
    x = skeleton.skeleton_points['RightHip'] - skeleton.skeleton_points['LeftHip']
    w = skeleton.skeleton_points['Torso'] - skeleton.skeleton_points['HipCenter']
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    hip_rot = np.vstack((x, y, z)).T
    ############################################################# 
    # Left foot
    x = skeleton.skeleton_points['LeftFootIndex'] - skeleton.skeleton_points['LeftHeel'] 
    w = skeleton.skeleton_points['LeftKnee'] - skeleton.skeleton_points['LeftHeel']
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    l_foot_rot = np.vstack((x, y, z)).T
    ############################################################# 
    # Right foot
    x = skeleton.skeleton_points['RightFootIndex'] - skeleton.skeleton_points['RightHeel'] 
    w = skeleton.skeleton_points['RightKnee'] - skeleton.skeleton_points['RightHeel']
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    r_foot_rot = np.vstack((x, y, z)).T
    
    hip_rot = R.from_matrix(hip_rot).as_quat()
    r_foot_rot = R.from_matrix(r_foot_rot).as_quat()
    l_foot_rot = R.from_matrix(l_foot_rot).as_quat()
    
    FootRot = FootRotation(RightFoot = r_foot_rot,
                           LeftFoot= l_foot_rot,
                           Hip = hip_rot)
    return FootRot

def get_rot(skeleton:Skeleton) -> FootRotation:
    ############################################################# 
    # Hip
    x = skeleton.skeleton_points['RightHip'] - skeleton.skeleton_points['LeftHip']
    w = skeleton.skeleton_points['Torso'] - skeleton.skeleton_points['HipCenter']
    z = np.cross(x, w)
    y = np.cross(z, x)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    hip_rot = np.vstack((x, y, z)).T

    #############################################################
    # Right leg 
    y = skeleton.skeleton_points['RightKnee'] - skeleton.skeleton_points['RightAnkle']
    w = skeleton.skeleton_points['RightHip'] - skeleton.skeleton_points['RightAnkle']
    z = np.cross(w, y)
    if np.sqrt(sum(z**2)) < 1e-6:
        w = skeleton.skeleton_points['LeftHip'] - skeleton.skeleton_points['RightAnkle']
        z = np.cross(w, y)
    x = np.cross(y,z)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    leg_r_rot = np.vstack((x, y, z)).T

    #############################################################
    # Left leg
    y = skeleton.skeleton_points['LeftKnee'] - skeleton.skeleton_points['LeftAnkle']
    w = skeleton.skeleton_points['LeftHip'] - skeleton.skeleton_points['LeftAnkle']
    z = np.cross(w, y)
    if np.sqrt(sum(z**2)) < 1e-6:
        w = skeleton.skeleton_points['RightHip'] - skeleton.skeleton_points['LeftAnkle']
        z = np.cross(w, y)
    x = np.cross(y,z)
    
    x = x/np.sqrt(sum(x**2))
    y = y/np.sqrt(sum(y**2))
    z = z/np.sqrt(sum(z**2))
    
    leg_l_rot = np.vstack((x, y, z)).T

    rot_hip = R.from_matrix(hip_rot).as_quat()
    rot_leg_r = R.from_matrix(leg_r_rot).as_quat()
    rot_leg_l = R.from_matrix(leg_l_rot).as_quat()
    
    FootRot = FootRotation(RightFoot = rot_leg_r,
                           LeftFoot  = rot_leg_l,
                           Hip       = rot_hip)
    return FootRot


def sendToPipe(text):
    if platform.startswith('win32'):
        pipe = open(r'\\.\pipe\ApriltagPipeIn', 'rb+', buffering=0)
        some_data = str.encode(text)
        some_data += b'\0'
        pipe.write(some_data)
        resp = pipe.read(1024)
        pipe.close()
    elif platform.startswith('linux'):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.connect("/tmp/ApriltagPipeIn")
        some_data = text.encode('utf-8')
        some_data += b'\0'
        client.send(some_data)
        resp = client.recv(1024)
        client.close()
    else:
        print(f"Unsuported platform {sys.platform}")
        raise Exception
    return resp

def sendToSteamVR_(text):
    #Function to send a string to my steamvr driver through a named pipe.
    #open pipe -> send string -> read string -> close pipe
    #sometimes, something along that pipeline fails for no reason, which is why the try catch is needed.
    #returns an array containing the values returned by the driver.
    try:
        resp = sendToPipe(text)
    except:
        return ["error"]

    string = resp.decode("utf-8")
    array = string.split(" ")
    
    return array


def sendToSteamVR(text, num_tries=10, wait_time=0.01):
    # wrapped function sendToSteamVR that detects failed connections
    ret = sendToSteamVR_(text)
    i = 0
    while "error" in ret:
        if i > 5:
            print("INFO: Error while connecting to SteamVR. Retrying...")
        time.sleep(wait_time)
        ret = sendToSteamVR_(text)
        i += 1
        if i >= num_tries:
            return None # probably better to throw error here and exit the program (assert?)
    
    return ret

    
class CameraStream():
    def __init__(self, params:'Parameters'):
        self.params = params
        self.image_ready = False
        # setup camera capture
        if len(params.cameraid) <= 2:
            cameraid = int(params.cameraid)
        else:
            cameraid = params.cameraid
            
        if params.camera_settings: # use advanced settings
            self.cap = cv2.VideoCapture(cameraid, cv2.CAP_DSHOW) 
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        else:
            self.cap = cv2.VideoCapture(cameraid)  

        if not self.cap.isOpened():
            print("ERROR: Could not open camera, try another id/IP")
            shutdown(params)

        if params.camera_height != 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(params.camera_height))
            
        if params.camera_width != 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(params.camera_width))

        print("INFO: Start camera thread")
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    
    def update(self):
        # continuously grab images
        while True:
            ret, self.image_from_thread = self.cap.read()    
            self.image_ready = True
            
            if ret == 0:
                print("ERROR: Camera capture failed! missed frames.")
                self.params.exit_ready = True
                return
 

def shutdown(params:'Parameters'):
    # first save parameters 
    print("INFO: Saving parameters...")
    params.save_params()

    cv2.destroyAllWindows()
    # sys.exit("INFO: Exiting... You can close the window after 10 seconds.")
