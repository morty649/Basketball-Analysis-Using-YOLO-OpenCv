from ultralytics import YOLO
import supervision as sv
import sys
import os
sys.path.append("../")
from utils import read_stub,save_stub

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch_frames,conf = 0.5)
            detections+=batch_detections
        return detections
    
    def get_object_tracks(self,frames,read_from_stub=False,stub_path=None):

        tracks = read_stub(read_from_stub,stub_path)  #check point
        if tracks is not None:
            if len(tracks)==len(frames):
                return tracks
            
        detections = self.detect_frames(frames)
        tracks = []

        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            tracks.append({}) #this is to capture features of ball
            #in a frame if two bbox are considered as ball then the one with high confidence is ball
            chosen_bbox=None
            max_conf =0

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                if max_conf<confidence:
                    chosen_bbox = bbox
                    max_conf = confidence

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox":chosen_bbox}

        save_stub(stub_path,tracks)
        return tracks