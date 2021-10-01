import numpy as np
import cv2
import os
import sys

class roi():
    def __init__(self, types=['right']):
        LBFmodel_file = "data/LFBmodel.yaml"
        self.landmark_detector  = cv2.face.createFacemarkLBF()
        
        old_stdout = sys.stdout # backup current stdout
        sys.stdout = open(os.devnull, "w")
        self.landmark_detector.loadModel(LBFmodel_file)
        sys.stdout = old_stdout # reset old stdout
        
        self.types = []
        for type in types:
            if type in ['left', 'right', 'forehead']:
                self.types.append(type)
                
            elif type == 'all':
                self.types = ['left', 'right', 'forehead']
                break
            
            else:
                raise ValueError("type should one of the following: 'right', 'left', 'forehead' or 'all'")
        
        
    def get_roi(self, frame, bbox):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, landmarks = self.landmark_detector.fit(gray, np.array([bbox]))
        landmark = landmarks[0]
        rois = []
        
        for type in self.types:
            if type == 'right':
                x = landmark[0,43,0]
                y = landmark[0,29,1]
                w = landmark[0,26,0] - landmark[0,43,0]
                h = landmark[0,30,1] - landmark[0,27,1]
                
            elif type == 'left':
                x = landmark[0,17,0]
                y = landmark[0,29,1]
                w = landmark[0,38,0] - landmark[0,17,0]
                h = landmark[0,30,1] - landmark[0,27,1]
                
            elif type == 'forehead':
                w = landmark[0,23,0] - landmark[0,20,0]
                h = landmark[0,29,1] - landmark[0,27,1]
                x = landmark[0,20,0]
                y = landmark[0,19,1] - h
                
                
            else:
                raise ValueError("type should one of the following: 'right', 'left', or 'forehead'")
            
            rois.append(((x-bbox[0])/bbox[2], (y-bbox[1])/bbox[3], w/bbox[2], h/bbox[3]))
        
        return rois
    
    
if __name__ == '__main__':
    obj = roi()