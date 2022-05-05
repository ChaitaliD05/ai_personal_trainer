#Import Libraries
import numpy as np
import time
import cv2
import mediapipe as mp
import math

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=True, trackCon=True):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True): #process the pose of human
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True): #find the position? circles and line 
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True): #angle detection
        # Get the landmarks, X Y , Z---- Angular points(red)
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        
        if angle < 0:
            angle += 360   # if angle i negative we use this line to make it positivE
            print("Angle",angle)
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)  #for lines b/w x and y
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED) # for inner circle
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2) #for outer circle
            cv2.circle(img, (x3, y3), 5, (0, 0,255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 10, y2 + 20),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 120, 0), 3)
        return angle

def main(): #make video more precise and accurate
    cap=cv2.VideoCapture(0)
    #cap=cv2.VideoCapture("C://Users//Admin//Desktop//p2.mp4")
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (720, 720))
        img = detector.findPose(img,draw=True)
        lmList = detector.findPosition(img, draw=False)
        #print(lmList)
        if len(lmList) != 0:
            print(lmList[22])
            cv2.circle(img, (lmList[22][1], lmList[22][2]), 10, (130, 255, 255), cv2.FILLED) 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        cv2.imshow("Image", img)
        key=cv2.waitKey(1)
        if key == ord("q") or key == ord("Q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()