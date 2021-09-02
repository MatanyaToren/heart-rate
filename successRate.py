import cv2
import sys
import os


if __name__ == '__main__':
    # Face Detectioin
    cascPath = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_alt.xml')
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read video
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # video = cv2.VideoCapture('videos/56bpm_17_08.mp4')
    # video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    for i in range(5):
        ok, frame = video.read()
        # print(frame.shape)
        if not ok:
            print('Cannot read video file')
            sys.exit()
    
    successes = 0
    N = 600

    for __ in range(N):
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        
        # Start timer
        timer = cv2.getTickCount()

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=6,
                    minSize=(100, 100),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            bbox = faces[0]
            successes += 1
        except IndexError:
            continue
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        # cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'): 
            cv2.destroyAllWindows()
            break

print('success rate is {:.2f}'.format(successes/N))