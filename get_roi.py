import numpy as np
import cv2
import os
import sys

class roi():
    def __init__(self, type='left'):
        LBFmodel_file = "data/LFBmodel.yaml"
        self.landmark_detector  = cv2.face.createFacemarkLBF()
        
        old_stdout = sys.stdout # backup current stdout
        sys.stdout = open(os.devnull, "w")
        self.landmark_detector.loadModel(LBFmodel_file)
        sys.stdout = old_stdout # reset old stdout
        
        
    def get_roi(self, frame, bbox):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, landmarks = self.landmark_detector.fit(gray, np.array([bbox]))
        landmark = landmarks[0]
        x = landmark[0,42,0]
        y = landmark[0,29,1]
        w = landmark[0,26,0] - landmark[0,42,0]
        h = landmark[0,30,1] - landmark[0,27,1]
        
        return ((x-bbox[0])/bbox[2], (y-bbox[1])/bbox[3], w/bbox[2], h/bbox[3])
    
    
if __name__ == '__main__':
    obj = roi()