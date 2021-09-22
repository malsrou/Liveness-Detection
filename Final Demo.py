from Baseline_VGG19_min import CreateBaselineModel
from mymodel_optimized import CreateMyModel
from tensorflow.keras.optimizers import RMSprop
from logging import NullHandler
import tensorflow as tf
import numpy as np
import cv2
from mtcnn import MTCNN
import json
from keras.preprocessing import image

class VideoCam():
    def __init__(self, url=0):
        self.url = url
        self.cap = cv2.VideoCapture(self.url)
        self.get_frame()
        self.get_frame_read()

    def show_frame(self, frame, name_fr='NAME'):
        cv2.imshow(name_fr, frame)
        cv2.waitKey(1)

    def get_frame(self):
        return self.cap.retrieve()

    def get_frame_read(self):
        return self.cap.read()

    def close_cam(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def restart_capture(self, cap):
        cap.release()
        self.cap = cv2.VideoCapture(self.url)

#Prepare Camera
SKIPFRAME = 3
url = 0
v1 = VideoCam(url)
ct = 0

#Prepare Models 
modelF= CreateMyModel()
model = CreateBaselineModel()

model.load_weights("weights_baselineVGG19_min_best.hdf5")
lr = 0.0001
model.compile(loss="binary_crossentropy", optimizer=RMSprop(learning_rate=lr), metrics=["accuracy"])

modelF.load_weights("weights_Mymodel_Opt_Face_Only_best copy.hdf5")
lr = 0.001
modelF.compile(loss="binary_crossentropy", optimizer=RMSprop(learning_rate=lr), metrics=["accuracy"])

#Prepare Face Detector - Fast Implementation 
detector = MTCNN(
            scale_factor=0.5,
            min_face_size=30)

#Prepare required variables as per defined algorithm
P1 =0
P2 =0
P = 0

PT = 0.4 
#PT=0.2
GDT = 8 
Global_Detector = 0
Current_State = 0

#for testing
i=0

while True:
    ct += 1
    try:
        ret = v1.cap.grab()
        if ct % SKIPFRAME == 0:  # skip some frames
            ret, frame = v1.get_frame()
            if not ret:
                v1.restart_capture(v1.cap)
                continue
            
            #cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2_im = cv2.resize(frame,(256,256), interpolation = cv2.INTER_AREA)
            #enable for testing
            #cv2.imwrite("C:/Users/malsr/Coding/ML Zaka - Project/Processed Data/Scene Testing/" + str(i) + ".png", cv2_im)
            #i=i+1  
            target_im = image.img_to_array(cv2_im)
            target_im = target_im/255 #normalize it

            target_im = np.expand_dims(target_im, axis=0)
            P1 = model.predict(target_im)
            
            print("P1 is " + str(P1))

            result = detector.detect_faces(frame)

            print(result, result.count("box"))

            if(result):
                if(result[0]==result[0]):
                    bounding_box = result[0]["box"]
                    start = (bounding_box[0],bounding_box[1])
                    end = (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3])
                    
                    face_frame = frame[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
                    img_face = cv2.resize(face_frame,(64,64), interpolation = cv2.INTER_AREA)
                    #enable for testing
                    #cv2.imwrite("C:/Users/malsr/Coding/ML Zaka - Project/Processed Data/Testing Cropping/" + str(i) + ".png", img_face)
                    #i=i+1
                    target_face = image.img_to_array(img_face)
                    target_face = target_face/255 #normalize it

                    target_face = np.expand_dims(target_face, axis=0)

                    P2 = modelF.predict(target_face)

                    print("P2 is " + str(P2))

                    frame = cv2.rectangle(frame, start, end, (255,0,0), 2)
                    frame = cv2.putText(frame, text=str(result[0]["confidence"]), org=(bounding_box[0]+bounding_box[2],bounding_box[1]),
                    fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(256,0,0),
                    thickness=2, lineType=cv2.LINE_AA)

                    P = ((1-P1) + P2)/2
                    #P=P2
                    print(P)

                    if (P <= PT):
                        P=0
                    else:
                        P=1
                    
                    print("P is " + str(P))

                    if(Current_State == P):
                        Global_Detector = Global_Detector + 1
                    else:
                        Current_State = P
                        Global_Detector = 0
                    
                    if (P==0):
                        frame = cv2.putText(frame, text="REAL", org=(bounding_box[0]+10,bounding_box[1]-50),
                        fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(256,0,0),
                        thickness=2, lineType=cv2.LINE_AA)
                    else:
                        frame = cv2.putText(frame, text="FAKE", org=(bounding_box[0]+10,bounding_box[1]-50),
                        fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(256,0,0),
                        thickness=2, lineType=cv2.LINE_AA)

                    if(Global_Detector>GDT):
                        if(Current_State == 0):
                            frame = cv2.putText(frame, text="Pass to FV", org=(bounding_box[0]-20,bounding_box[1]+75),
                            fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=4, color=(256,0,0),
                            thickness=2, lineType=cv2.LINE_AA)
                            
            v1.show_frame(frame, 'frame')

    except KeyboardInterrupt:
        v1.close_cam()
        exit(0)