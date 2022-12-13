import cv2
import numpy as np
from keras.models import model_from_json


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('files/fil.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("files/new_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
#cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

while True:
    _,frame = video_capture.read() 
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25) 
    rgb_small_frame = small_frame[:,:,::-1] 
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame) 
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = [] 
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding) 
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding) 
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]: 
                name = known_faces_names[best_match_index]

                face_names.append(name)
                if name in known_faces_names: 
                    if name in students:
                        students.remove(name)
                        print(students)
                        current_time = now.strftime("%H-%M-%S") 
                        lnwriter.writerow([name, current_time])
                        
    cv2.imshow("attendence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
                        
video_capture.release() 
cv2.destroyAllWindows() 
f.close()