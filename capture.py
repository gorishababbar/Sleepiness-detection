import face_recognition
import cv2

IP_Webcam = False
flag = False

if IP_Webcam is True:
    video_capture = cv2.VideoCapture('http://192.168.1.100:8080/videofeed')  # IP Webcam
else:
    video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    cv2.imshow('Video', frame)
    flag = False

    c = cv2.waitKey(1)
    if 'q' == chr(c & 255):
        print("Exited Operation !!")
        exit()

    if 's' == chr(c & 255):
        flag = True
        break

if IP_Webcam is not True:
    video_capture.release()
cv2.destroyAllWindows()

if flag:
    print("Processing captured image...")

    # Add your face recognition and drowsiness detection logic here

    print("Processing completed!")
