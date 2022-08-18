#I want to create a generic Hand tracking module using opencv and mediapipe framework

#Useful libraries for working with Computer Vision -->
#Python libraries 
import numpy as np
import matplotlib.pyplot as plt 
import datetime
import time
import math

#Computer Vision libraries
import cv2 as cv
import mediapipe as mp



print("HAND TRACKING MODULE \n")

class handDetector():
    def __init__(self, staticImageMode = False, maxHands = 2, complexity = 1, minDetectionConfidence = 0.5, minTrackingConfidence = 0.5): #The mode is false because we want it to alternate between tracking and detecting, it only detects if its set to True 
        self.staticImageMode = staticImageMode
        self.maxHands = maxHands
        self.complexity = complexity
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        #We are instantiating the objects of hand tracking functions
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(staticImageMode, maxHands, complexity, minDetectionConfidence, minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20] #This reprensent landmarks of the finger tips
    
    #This function handles the various hand landmarks in the frame
    def findHands(self, image, draw = True):
        rgbImage = cv.cvtColor(image, cv.COLOR_BGR2RGB) #We are converting the format because mediapipe works with RGB format
        global results
        results = self.hands.process(rgbImage) #This gets the various points in the hands
        #print(f"Results: {results}")
        #print(f"Result landmarks length: {results.multi_hand_landmarks} \n")

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS) #This gets the drawings in the hands
                else:
                    pass

        return image

    #This function allows us to find specific positions in the hand
    def findPosition(self, image, handNo = 0, draw = True): #, handNo = 8
        global yList, xList, bbox, lmList
        yList = []
        xList = []
        bbox = []
        lmList = []
        if results.multi_hand_landmarks:
            #print(f" -----------------Start: {len(results.multi_hand_landmarks)}--------------------")
            hands = results.multi_hand_landmarks[handNo]
            #print(f"Hands: {hands}")
            #print(f"Hands landmarks: {hands.landmark}")
            #for hands in results.multi_hand_landmarks: #[handNo]
            for id, lm in enumerate(hands.landmark): #.landmark
                #if id == handNo:  
                h, w, c = image.shape
                #print(f"Landmark: {id}: {lm} -------------------------------->")
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(f"Landmarks: {id}: CX: {cx}, CY: {cy} \n")
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy]) #This appends the various points to the landmark list
                        
                #This draws the circle on the selected landmark
                if draw:
                    cv.circle(image, (cx, cy), 10, (255, 0, 0), cv.FILLED)       
            
            #This helps us get the dimensions of the bounding box for the detected hands
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            bbox = xMin, yMin, xMax, yMax
            #print(f"BBOX: {bbox}")
            if draw:
                cv.rectangle(image, (xMin - 20, yMin - 20), (xMax + 20, yMax + 20), (0, 255, 0), 2)

        #print(f"X List: {xList}")
        #print(f"Length: {len(lmList)}, Landmark List: {lmList}")
        return lmList, bbox

    #This function helps us to check for which fingers are up
    def fingersUp(self, lmList, tipIds, flipped = True):
        fingers = []

        #Check for the location of the thumb
        if flipped: #Checking if the screen is flipped 
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        elif not flipped:
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            pass

        #Remaining fingers 
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0) 
            totalFingers = fingers.count(1)
            #print(f"Total Fingers: {totalFingers}")
        return fingers, totalFingers

    #This function helps to find distance between two finger tips
    def findDistance(self, point1, point2, image, draw = True, r = 15, t = 3):
        x1, y1 = lmList[0][point1][1:]
        x2, y2 = lmList[0][point2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 #This helps find the distance between the points
        
        if draw:
            cv.line(image, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(image, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(image, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(image, (cx, cy), r, (0, 0, 255), cv.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, image, [x1, y1, x2, y2, cx, cy]



def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #print(f"Frame Size: {frame.shape}")
            #print(f"Frame Size: {frame.shape[0]}")
            frame = detector.findHands(frame)
            lmList, bbox = detector.findPosition(frame, draw = False)
            for i in range(10):
                if len(lmList) != 0:
                    pass
                    #print(f"Landmark List: {lmList} \n")
                    #print(f"1st item Landmark List: {lmList[4]} \n")
                    ##print(f"1st of 1st item Landmark List: {lmList[0][0:1][0]} \n")
                    #print(f"1st of 1st item Landmark List: {lmList[4][2]} \n")
                    fingers = detector.fingersUp(lmList, [4, 8, 12, 16, 20])
                    #print(f"Fingers: {fingers}")

            #Getting the frames per second
            cTime = time.perf_counter()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv.putText(frame, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv.imshow("Frame", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()




print("Finished Executing!!!")