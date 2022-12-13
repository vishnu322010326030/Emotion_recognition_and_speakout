import cv2
import numpy as np
import face_recognition
import cv2
import threading
import numpy as np
import csv
import os
from datetime import datetime
from keras.models import model_from_json
from win32com.client import Dispatch
__lock = threading.Lock()

def speak(str):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.speak(str)


emotion_dict = {0: "Happy", 1: "kishore", 2: "Neutral", 3: "sad", }

# load json and create model
json_file = open('D:/ana/sentimental project/NEW_MODEL/files/kishore_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("D:/ana/sentimental project/NEW_MODEL/files/kishore_model1.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)



known_faces_names = [
"Happy",
"kishore",
"Neutral",
"Sad"
]

students = known_faces_names.copy()
face_locations = [] 
face_encodings =[]
face_names = []

s=True

now = datetime.now() 
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    # find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920, 1080))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    se = set()

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        
        name=""
        name = emotion_dict[maxindex]
        face_names.append(name)
        if emotion_dict[maxindex] in emotion_dict.values():
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S") 
                    lnwriter.writerow([name, current_time])
                    se.add(name)
            
            
    if "kishore" in se:
        speak("Good morning")


    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






  


                        
  
                        
