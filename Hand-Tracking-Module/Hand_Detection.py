import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode =False, maxhand = 2, detectioncon = 0.5, trackingcon =0.5):
        self.mode = mode
        self.maxhand = maxhand
        self.detectioncon = detectioncon
        self.trackingcon = trackingcon
        self.tipid =[4,8,12,16,20]

        self.mphands = mp.solutions.hands

        self.hands  = self.mphands.Hands(self.mode, self.maxhand, self.detectioncon, self.trackingcon)
        
        self.mpdraw = mp.solutions.drawing_utils

    def Findhands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results)

        if self.results.multi_hand_landmarks:
            if draw:
                for handlm in self.results.multi_hand_landmarks:
                    self.mpdraw.draw_landmarks(img, handlm, self.mphands.HAND_CONNECTIONS)
        return img
    
    def Findloaction(self, img, handNo = 0, draw = True):
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            handlm = self.results.multi_hand_landmarks[handNo]
            for id , lm in enumerate(handlm.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w) , int(lm.y * h)
                # print(id, cx, cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmlist
    
    def Finger_UP(self,Left=True,Right =False):
        
        finger = []
        if Right:
            if self.lmlist[self.tipid[0]][1] < self.lmlist[self.tipid[0]-1][1]:
                finger.append(1)
            else:
                finger.append(0)
        if Left:
            if self.lmlist[self.tipid[0]][1] > self.lmlist[self.tipid[0]-1][1]:
                finger.append(1)
            else:
                finger.append(0)

        for id in range(1,5):
            if self.lmlist[self.tipid[id]][2] < self.lmlist[self.tipid[id]-2][2]:
                finger.append(1)
            else:
                finger.append(0)
        return finger

def main():
    cap  = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        suc,img = cap.read()
        detector.Findhands(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
    