import cv2
import numpy as np
import dlib
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

croppable = False
fps = 10

# Save up to 3 faces to freeze. Format as (crop, mask, velocity)
freeze_face = []

# Save up to 2 history central points to determine velocity. Format as (x, y, time)
history_central = []


def overlay_face(dt):
    for i in range(len(freeze_face)):
        crop, mask, v = freeze_face[i]
        real[mask == 255] = crop[mask == 255]
        # shift crop and mask
        dx = v[0] * dt
        dy = v[1] * dt
        T = np.float32([[1, 0, np.ceil(dx)], [0, 1, np.ceil(dy)]])
        rows, cols = freeze_face[i][1].shape
        freeze_face[i][1] = cv2.warpAffine(freeze_face[i][1], T, dsize=(cols, rows))
        freeze_face[i][0] = cv2.warpAffine(freeze_face[i][0], T, dsize=(cols, rows))


def add_freeze_face(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and croppable:
        crop = np.zeros_like(frame)
        crop[mask == 255] = frame[mask == 255]
        x0, y0, t0 = history_central[0]
        x1, y1, t1 = history_central[1]
        dt = t1 - t0
        v = ((x1 - x0) / dt, (y1 - y0) / dt)

        freeze_face.append([crop, mask, v])
        # print("added face with velocity:", v)
        if len(freeze_face) > 3:
            freeze_face.pop(0)


def find_face_and_mouth(frame, real, landmarks):
    face_outline = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17)]
    face_outline.extend([(landmarks.part(i).x, landmarks.part(i).y) for i in range(26, 16, -1)])
    face_outline = np.array(face_outline)

    mouth_outline = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)])

    history_central.append((landmarks.part(30).x, landmarks.part(30).y, time.time()))
    if len(history_central) > 2:
        history_central.pop(0)

    cv2.drawContours(frame, [face_outline, mouth_outline], -1, (255, 255, 255), 1)
    cv2.drawContours(real, [face_outline, mouth_outline], -1, (255, 255, 255), 1)

    for i in range(60):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(frame, (x, y), 2, (255, 255, 255), 1)
        cv2.circle(real, (x, y), 2, (255, 255, 255), 1)

    return [face_outline, mouth_outline]


def gen_mask(gray, face_outline):
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [face_outline], -1, 255, -1)
    return mask






cap = cv2.VideoCapture(0)

cv2.namedWindow("Crop Your Face")
cv2.setMouseCallback("Crop Your Face", add_freeze_face)

while cap.isOpened():
    t = time.time()
    ret, frame = cap.read()
    real = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        croppable = False

    for face in faces:
        landmarks = predictor(gray, face)

        if landmarks is not None:
            face_outline, mouth_outline = find_face_and_mouth(frame, real, landmarks)
            mask = gen_mask(gray, face_outline)

            croppable = True

    overlay_face(1 / fps)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Crop Your Face", real)
    fps = 1/(time.time() - t)
    # print("fps: ", fps, end='\r')

    
cap.release()
cv2.destroyAllWindows()

