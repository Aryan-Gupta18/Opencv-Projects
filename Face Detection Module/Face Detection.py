import cv2
import mediapipe as mp
import time

class Face_Detector():
    def __init__(self, mdeccon = 0.5):
        self.mdeccon = mdeccon
        self.mpface = mp.solutions.mediapipe.python.solutions.face_detection
        self.faces = self.mpface.FaceDetection()
        self.draw = mp.solutions.drawing_utils

    def Find_Faces(self, img):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.faces.process(imgrgb)
        # print(result)
        if result.detections:
            for id , dector in enumerate(result.detections):
                bboxc = dector.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxc.xmin * w), int(bboxc.ymin * h), int(bboxc.width * w) , int(bboxc.height * h)
                img = self.Facny_Draw(img, bbox)
                cv2.putText(img,f'{int(dector.score[0] * 100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2,(0,255, 0), 4)
    
    def Facny_Draw(self, img, bbox, l=25, t=5):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bbox, (255, 0, 255), 1)
        # top left corner
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # top right corner
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # bottom left corner
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)
        # # bottom right corner
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)
        return img
    
def main():
    cap =  cv2.VideoCapture(0)
    face_detector = Face_Detector()
    while True:
        suc, img = cap.read()
        face_detector.Find_Faces(img)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()