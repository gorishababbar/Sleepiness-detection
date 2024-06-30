import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist
import sqlite3
import pygame

IP_Webcam = False

# Initialize Pygame mixer
pygame.mixer.init()

# Load MP3 file for alarm
pygame.mixer.music.load("Willow.mp3")

db = sqlite3.connect('db.sqlite3')
print("Opened Database Successfully !!")

cursor = db.cursor()

cursor.execute("SELECT * FROM sqlite_master WHERE name ='FACES' and type='table';")
chk = cursor.fetchone()
if chk is not None:
    data = cursor.execute("SELECT FACE_NAME, FACE_ENCODING FROM FACES")
else:
    print("There's no face entry in the Database !!")
    exit()

known_face_names = []
known_face_encodings = []

for row in data:
    known_face_names.append(row[0])
    known_face_encodings.append(np.frombuffer(row[1]))

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the video capture object
if IP_Webcam:
    video_capture = cv2.VideoCapture('http://192.168.1.100:8080/videofeed')  # IP Webcam
else:
    video_capture = cv2.VideoCapture(0)

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

def detect_drowsiness(ear, thresh):
    if ear < thresh:
        return True
    else:
        return False

thresh = 0.25 # Adjust this threshold as needed

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector(gray, 0)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []

        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))

        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        drowsy = detect_drowsiness(avg_ear, thresh)

        if drowsy:
            # Trigger alarm when drowsiness is detected
            pygame.mixer.music.play()

        # Draw eye regions for visualization
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, f'EAR: {avg_ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exited Operation !!")
        break

if IP_Webcam is not True:
    video_capture.release()
cv2.destroyAllWindows()
