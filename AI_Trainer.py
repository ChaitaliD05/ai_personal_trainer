"IMPORTING THE LIBRARIES"
import posemodule as pm
from flask import Flask, render_template, Response, request, redirect, url_for
import numpy as np
import cv2                              


app = Flask(__name__)


@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/show')
def plot():
    #cap=cv2.VideoCapture("C://Users//Admin//Desktop//p2.mp4")
    cap=cv2.VideoCapture(0)
    detector=pm.poseDetector()
    count = 0
    dir = 0
    while True: 
        success, img=cap.read()
        img=cv2.resize(img,(1000,720))
        img=detector.findPose(img,False) #for draw line
        lmList=detector.findPosition(img,False) #for creating dots
        
        if len(lmList)!=0:
            #These points is for left arm
            angle=detector.findAngle(img,11,13,15)
            #for percentage
            per = np.interp(angle, (210,310), (0, 100)) #2nd shows the  to 100%
            print("PERCENT LEVEL",per)

            # Check for the Angle and give colour
            bar = np.interp(angle, (220, 310), (400, 100))  #2nd shows bar height n width
            color = (255, 0, 255)
    
            if per == 100:
                color = (0, 255, 0)
                if dir == 0: # upper move
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 0, 255)
                if dir == 1:
                    count += 0.5
                    dir = 0
            print("SETS COUNTS",count)
            # Draw Bar
            cv2.rectangle(img, (250, 100), (200, 400), color, 2)
            cv2.rectangle(img, (250, int(bar)), (200, 400), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (210, 65), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)
            #DRAW hand CURLS
            cv2.rectangle(img, (0, 550), (100, 700), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (20, 620), cv2.FONT_HERSHEY_PLAIN, 5,
                        (255.0,0), 5)
    
        cv2.imshow("WEBCAM",img)
    
        key=cv2.waitKey(1)
        if key == ord("q") or key == ord("Q"):
            break
    
    cap.release()
    cv2.destroyAllWindows() #realesing the webcam
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False,port=5000)