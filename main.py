import cv2
import numpy as np
import dlib
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

croppable = False

freeze_face = []

def overlay_face(frame, faces):
    for crop, mask in faces:
        real[mask == 255] = crop[mask == 255]

def add_freeze_face(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and croppable:
        crop = np.zeros_like(frame)
        crop[mask == 255] = frame[mask == 255]
        freeze_face.append((crop, mask))
        print("crop!")
        if len(freeze_face) > 3:
            freeze_face.pop(0)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Crop Your Face")
cv2.setMouseCallback("Crop Your Face", add_freeze_face)

while cap.isOpened():
    # t = time.time()
    ret, frame = cap.read()
    real = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        croppable = False

    for face in faces:
        landmarks = predictor(gray, face)
        if landmarks is not None:
            face_outline = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17)]
            face_outline.extend([(landmarks.part(i).x, landmarks.part(i).y) for i in range(26, 16, -1)])
            face_outline = np.array(face_outline)

            mouth_outline = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)])

            cv2.drawContours(real, [face_outline, mouth_outline], -1, (255, 255, 255), 1)

            for i in range(60):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(real, (x, y), 2, (255, 255, 255), 1)

            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [face_outline], -1, 255, -1)
            croppable = True

    overlay_face(frame, freeze_face)
        

        


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

    cv2.imshow("Crop Your Face", real)
    # print("fps: ", 1/(time.time() - t), end='\r')

    
cap.release()
cv2.destroyAllWindows()

