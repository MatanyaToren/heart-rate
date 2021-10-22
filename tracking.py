import numpy as np
import cv2


class OutOfFrameError(RuntimeError):
    def __init__(self):
        self.message = 'Face is partly out of frame, please move to the center'

class TrackingError(RuntimeError):
    def __init__(self):
        self.message = 'Tracking failure detected'

class DetectionError(RuntimeError):
    def __init__(self):
        self.message = "face is not detected, can't initialize tracker"

class JumpingError(RuntimeError):
    def __init__(self):
        self.message = 'Face detection jumped'


class FaceTracker():
    """
    Face tracker combining V&J face detection and one of the trackers implemented in open-cv
    """
    def __init__(self, detectionRate=120):
        # Viola & Jones face recognition
        cascPath = '.\data\haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(cascPath)

        # face re-detection rate
        self.detectionRate = detectionRate
        self.detecionCounter = 0
        self.lastDetection = False

        # short term tracking
        self.tracker = None # cv2.legacy.TrackerMOSSE_create()

        # bounding box of the face, in the format (x,y,w,h)
        self.bbox = (0,0,0,0)


    def detect(self, frame):
        """
        detect face using V&J
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

        try:
            # dis = []
            # for face in faces:
            #     dis.append(np.square((face[0]-self.bbox[0])**2 + (face[1]-self.bbox[1])**2))

            # self.bbox = faces[np.argmin(dis)]
            self.bbox = faces[0]
            # if np.min(dis) / np.square(frame.shape[0]*frame.shape[1]) > 0.1 :
            #     # raise JumpingError
            #     pass

            # need to check if bbox boundaries are within frame
            self.checkBB(frame.shape[:2])
            self.lastDetection = True
            return self.bbox
        
        except (IndexError, ValueError):
            raise DetectionError()


    def initTracker(self, frame):
        """
        initialize short term tracker based on last detection
        """
        if self.lastDetection is False:
            raise DetectionError()

        self.tracker = cv2.legacy.TrackerMOSSE_create()
        self.tracker.init(frame, tuple(self.bbox))
        self.detecionCounter = 0


    def track(self, frame):
        """
        track face based on previous detection
        """
        ok, bbox = self.tracker.update(frame)
        if ok:
            self.bbox = bbox
        else:
            self.lastDetection = False
            raise TrackingError()

        # need to check if bbox boundaries are within frame
        self.checkBB(frame.shape[:2])
        self.detecionCounter += 1


    def update(self, frame):
        """
        long term tracking using re-detection and tracking
        """

        if self.lastDetection is False or self.detecionCounter >= self.detectionRate: 
            self.detect(frame)
            self.initTracker(frame)
        
        else:
            self.track(frame)

        return self.bbox



    def checkBB(self, sizeFrame):
        """
        check if bbox boundaries are within frame
        """    
        x, y, w, h = self.bbox
        hight, width = sizeFrame

        if x+w >= width or y+h >= hight or x < 0 or y < 0:
            self.lastDetection = False
            # print('bbox out of frame')
            raise OutOfFrameError()
        


