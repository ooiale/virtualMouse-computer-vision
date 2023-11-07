import cv2
import numpy as np
import time
import handtrackingmodule as htm
import pyautogui

pyautogui.FAILSAFE = False

##########
wCam, hCam = 640, 480
smoothening = 8
plocX, plocY = 0, 0
clocX, clocY = 0, 0
##########


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

cTime = 0
pTime = 0
detector = htm.handDetector()

movingMode = False
clickingMode = False
handCenter = 0

while True:
    success, img = cap.read()
    if not success:
        break

    #1. find hand landmarks
    #2. index down = click
    #3. only index = moving mode

    detector.findHands(img, draw=True)
    lmList = detector.find_Position(img, draw=False)

    if len(lmList) != 0:
        fingers = detector.fingersUp()
        movingMode = False
        clickingMode = False

        if fingers == [0,1,0,0,0]:
            movingMode = True
        elif fingers == [0,1,1,0,0]:
            clickingMode = True

        handCenter = detector.handCenter(img , draw = False)
        indexTip = [lmList[8][1] , lmList[8][2]]

        cv2.rectangle(img, (100,100), (wCam-100, hCam-100), (0,0,0), 2)

        if movingMode:
            x = int(np.interp(indexTip[0] , (0+100, wCam-100), (0, 1920)))
            y = int(np.interp(indexTip[1], (0+100,hCam-100), (0,1080)))

            clocX = plocX + (x - plocX)/smoothening
            clocY = plocY + (y - plocX)/smoothening

            pyautogui.moveTo(1920 - x, y)

            plocX, plocY = clocX, clocY

            #print(indexTip[0], indexTip[1], x,y)

        if clickingMode:
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(0.5)

    
        



    cv2.imshow('image', img)
    cv2.waitKey(1)