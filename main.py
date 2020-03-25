import cv2
import numpy as np
import time

from hand_tracker import HandTracker
from hand_functions import *
from face_functions import *

PALM_MODEL_PATH = "./model/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "./model/hand_landmark.tflite"
ANCHORS_PATH = "./model/anchors.csv"
SHAPE_PREDICTOR_PATH = "./model/shape_predictor_68_face_landmarks.dat"

face_detector = FaceTracker(
    predictor_path=SHAPE_PREDICTOR_PATH
)

hand_detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)


croppable = False
fps = 10


"""
    Mouse call back.
    Store current face into detector.freeze_faces.
"""
def add_freeze_face(event, x, y, flags, detector):
    if event == cv2.EVENT_LBUTTONDOWN and croppable:
        # cut out face from frame
        crop = np.zeros_like(frame)
        crop[mask == 255] = frame[mask == 255]

        # calculate velocity: (vx, vy)
        x0, y0, t0 = detector.history_central[0]
        x1, y1, t1 = detector.history_central[1]
        dt = t1 - t0
        v = ((x1 - x0) / dt, (y1 - y0) / dt)

        detector.freeze_faces.append([crop, mask, v])
        if len(detector.freeze_faces) > 3:
            detector.freeze_faces.pop(0)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("Crop Your Face")
    cv2.setMouseCallback("Crop Your Face", add_freeze_face, face_detector)

    while cap.isOpened():
        t = time.time()
        ret, frame = cap.read()
        real = frame.copy()
        detect_hands_and_draw_skeleton(frame, real, hand_detector)

        # detect outlines
        face_mouth_pairs = face_detector.face_pipeline(frame, real)
        if len(face_mouth_pairs) == 0:
            croppable = False

        # get masks to crop face
        for face_outline, _ in face_mouth_pairs:
            mask = face_detector.gen_mask(frame, face_outline)
            croppable = True

        # overlay stored faces
        face_detector.overlay_face(real, 1 / fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow("Crop Your Face", real)
        fps = 1/(time.time() - t)
        # print("fps: ", fps, end='\r')

    cap.release()
    cv2.destroyAllWindows()
