import time
from abc import ABC, abstractmethod
from helpers import  sendToSteamVR
from scipy.spatial.transform import Rotation as R
from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder
from pythonosc import udp_client

from helpers import shutdown, Skeleton, FootRotation
from parameters import Parameters
import numpy as np

class Backend(ABC):

    @abstractmethod
    def onparamchanged(self, params):
        ...

    @abstractmethod
    def connect(self, params):
        ...

    @abstractmethod
    def updatepose(self, params, skeleton:Skeleton, rots:FootRotation, hand_rots):
        ...

    @abstractmethod
    def disconnect(self):
        ...

class DummyBackend(Backend):

    def __init__(self, **kwargs):
        pass

    def onparamchanged(self, params):
        pass

    def connect(self, params):
        pass

    def updatepose(self, params, skeleton:Skeleton, rots:FootRotation, hand_rots):
        pass

    def disconnect(self):
        pass

class SteamVRBackend(Backend):

    def __init__(self, **kwargs):
        pass

    def onparamchanged(self, params:Parameters):
        resp = sendToSteamVR(f"settings 50 {params.smoothing} {params.additional_smoothing}")
        if resp is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)
            
    def connect(self, params:Parameters):
        print("Connecting to SteamVR")

        #ask the driver, how many devices are connected to ensure we dont add additional trackers
        #in case we restart the program
        numtrackers = sendToSteamVR("numtrackers")
        if numtrackers is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)

        numtrackers = int(numtrackers[2])

        #games use 3 trackers, but we can also send the entire skeleton if we want to look at how it works
        roles = ["TrackerRole_Waist", "TrackerRole_RightFoot", "TrackerRole_LeftFoot"]
        if params.preview_skeleton:
            totaltrackers = 23
        else:
            totaltrackers = 3
            if params.use_hands:
                totaltrackers += 2
                roles.append("TrackerRole_Handed")
                roles.append("TrackerRole_Handed")
            if params.use_elbows:
                totaltrackers += 2
                roles.append("TrackerRole_RightElbow")
                roles.append("TrackerRole_LeftElbow")
            if params.use_knees:
                totaltrackers += 2
                roles.append("TrackerRole_RightKnee")
                roles.append("TrackerRole_LeftKnee")

            if params.ignore_hip:
                totaltrackers -= 1
                del roles[0]

        for i in range(len(roles),totaltrackers):
            roles.append("None")

        for i in range(numtrackers,totaltrackers):
            #sending addtracker to our driver will... add a tracker. to our driver.
            resp = sendToSteamVR(f"addtracker MPTracker{i} {roles[i]}")
            if resp is None:
                print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
                shutdown(params)

        resp = sendToSteamVR(f"settings 50 {params.smoothing} {params.additional_smoothing}")
        if resp is None:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)

    def updatepose(self, params:Parameters, skeleton:Skeleton, rots:FootRotation, hand_rots):
        HeadSetPosAndRot = sendToSteamVR("getdevicepose 0")        #get hmd data to allign our skeleton to

        if HeadSetPosAndRot is None or len(HeadSetPosAndRot) < 10:
            print("ERROR: Could not connect to SteamVR after 10 tries! Launch SteamVR and try again.")
            shutdown(params)

        headsetpos = [float(HeadSetPosAndRot[3]),float(HeadSetPosAndRot[4]),float(HeadSetPosAndRot[5])]
        headsetrot = R.from_quat([float(HeadSetPosAndRot[7]),float(HeadSetPosAndRot[8]),float(HeadSetPosAndRot[9]),float(HeadSetPosAndRot[6])])

        neckoffset = headsetrot.apply(params.hmd_to_neck_offset)   #the neck position seems to be the best point to allign to, as its well defined on
                                                            #the skeleton (unlike the eyes/nose, which jump around) and can be calculated from hmd.

        if params.recalibrate:
            print("INFO: frame to recalibrate")

        else:
            skeleton = skeleton * params.posescale     #rescale skeleton to calibrated height
            # offset = np.array([0.1, 0, 0])
            offset = skeleton.skeleton_points['Torso'] - (headsetpos+neckoffset)    #calculate the position of the skeleton
            if not params.preview_skeleton:
                joint = skeleton.skeleton_points['RightAnkle'] - offset
                sendToSteamVR(f"updatepose 1 {joint[0]} {joint[1]} {joint[2]} {rots.RightFoot[3]} {rots.RightFoot[0]} {rots.RightFoot[1]} {rots.RightFoot[2]} {params.camera_latency} 0.8")

                joint = skeleton.skeleton_points['LeftAnkle'] - offset
                sendToSteamVR(f"updatepose 2 {joint[0]} {joint[1]} {joint[2]} {rots.LeftFoot[3]} {rots.LeftFoot[0]} {rots.LeftFoot[1]} {rots.LeftFoot[2]} {params.camera_latency} 0.8")

                if not params.ignore_hip:
                    joint = skeleton.skeleton_points['HipCenter'] - offset
                    sendToSteamVR(f"updatepose 0 {joint[0]} {joint[1]} {joint[2]} {rots.Hip[3]} {rots.Hip[0]} {rots.Hip[1]} {rots.Hip[2]} {params.camera_latency} 0.8")

                if params.use_hands:
                    joint = skeleton.skeleton_points['RightWrist'] - offset
                    sendToSteamVR(f"updatepose 3 {joint[0]} {joint[1]} {joint[2]} {hand_rots[0][3]} {hand_rots[0][0]} {hand_rots[0][1]} {hand_rots[0][2]} {params.camera_latency} 0.8")

                    joint = skeleton.skeleton_points['LeftWrist'] - offset
                    sendToSteamVR(f"updatepose 4 {joint[0]} {joint[1]} {joint[2]} {hand_rots[1][3]} {hand_rots[1][0]} {hand_rots[1][1]} {hand_rots[1][2]} {params.camera_latency} 0.8")
                pass

            else:
                for i, (key, value) in zip(range(skeleton.Nr_of_skeleton_points), skeleton.skeleton_points.items()):
                    joint = value - offset      #if previewing skeleton, send the position of each keypoint to steamvr without rotation
                    sendToSteamVR(f"updatepose {i} {joint[0]} {joint[1]} {joint[2] - 2} 1 0 0 0 {params.camera_latency} 0.8")
        return True

    def disconnect(self):
        pass

def osc_build_msg(name, position_or_rotation, args):
    builder = osc_message_builder.OscMessageBuilder(address=f"/tracking/trackers/{name}/{position_or_rotation}")
    builder.add_arg(float(args[0]))
    builder.add_arg(float(args[1]))
    builder.add_arg(float(args[2]))
    return builder.build()

def osc_build_bundle(trackers):
    builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
    builder.add_content(osc_build_msg(trackers[0]['name'], "position", trackers[0]['position']))
    for tracker in trackers[1:]:
        builder.add_content(osc_build_msg(tracker['name'], "position", tracker['position']))
        builder.add_content(osc_build_msg(tracker['name'], "rotation", tracker['rotation']))
    return builder.build()

class VRChatOSCBackend(Backend):

    def __init__(self, **kwargs):
        self.prev_pose3d = np.zeros((29,3))
        pass

    def onparamchanged(self, params):
        pass

    def connect(self, params):
        if hasattr(params, "backend_ip") and hasattr(params, "backend_port"):
            self.client = udp_client.UDPClient(params.backend_ip, params.backend_port)
        else:
            self.client = udp_client.UDPClient("127.0.0.1", 9000)

    def updatepose(self, params, pose3d:Skeleton, rots, hand_rots):
    
        #pose3d[:,1] = -pose3d[:,1]      #flip the positions as coordinate system is different from steamvr
        #pose3d[:,0] = -pose3d[:,0]
        
        pose3d = self.prev_pose3d*params.additional_smoothing + pose3d*(1-params.additional_smoothing)
        self.prev_pose3d = pose3d
    
        headsetpos = [float(0),float(0),float(0)]
        headsetrot = R.from_quat([float(0),float(0),float(0),float(1)])

        neckoffset = headsetrot.apply(params.hmd_to_neck_offset)   #the neck position seems to be the best point to allign to, as its well defined on
                                                            #the skeleton (unlike the eyes/nose, which jump around) and can be calculated from hmd.
        if params.recalibrate:
            print("frame to recalibrate")
        else:
            pose3d = pose3d * params.posescale     #rescale skeleton to calibrated height
            #print(pose3d)
            offset = pose3d[7] - (headsetpos+neckoffset)    #calculate the position of the skeleton
            if not params.preview_skeleton:
                trackers = []
                trackers.append({ "name": "head", "position": [ 0, 0, 0 ]})
                if not params.ignore_hip:
                    for i in [(0,1),(5,2),(6,0)]:
                        #left foot should be position 0 and rotation 1, but for osc, the rotations got switched at some point so its (0,2)
                        position = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        #position[0] = -position[0]
                        #position[1] = -position[1]
                        position[2] = -position[2]
                        rotation = R.from_quat(rots[i[1]])
                        #rotation *= R.from_euler("ZY", [ 180, -90 ], degrees=True)
                        rotation = rotation.as_euler("zxy", degrees=True)
                        rotation = [ -rotation[1], -rotation[2], rotation[0] ]  #mirror the rotation, as we mirrored the positions
                        trackers.append({ "name": str(i[1]+1), "position": position, "rotation": rotation })
                else:
                    for i in [(0,1),(5,2)]:
                        position = pose3d[i[0]] - offset       #for each foot and hips, offset it by skeleton position and send to steamvr
                        #position[0] = -position[0]
                        #position[1] = -position[1]
                        position[2] = -position[2]
                        rotation = R.from_quat(rots[i[1]])
                        #rotation *= R.from_euler("ZY", [ 180, -90 ], degrees=True)
                        rotation = rotation.as_euler("zxy", degrees=True)
                        rotation = [ -rotation[1], -rotation[2], rotation[0] ]
                        trackers.append({ "name": str(i[1]+1), "position": position, "rotation": rotation })
                if params.use_hands:
                    # Sending hand trackers unsupported
                    pass
                if len(trackers) > 1:
                    self.client.send(osc_build_bundle(trackers))

            else:
                # Preview skeleton unsupported
                pass
        return True

    def disconnect(self):
        pass
