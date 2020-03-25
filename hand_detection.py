import cv2
import numpy as np
import dlib

from model.demo.common.mva19v2 import Estimator, preprocess

detector = dlib.fhog_object_detector("model/HandDetector.svm")
# predictor = dlib.shape_predictor("model/shape_predictor_9_hand_landmarks.dat")

NPOINTS = 9
# PAIRS = [[8, 1], [8, 3], [8, 5], [8, 6], [8, 7], [0, 1], [2, 3], [4, 5]]
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]


box_size = 224
stride = 4
THRESHOLD = 0.5

model_file = "./model/demo/models/mobnet4f_cmu_adadelta_t1_model.pb"
input_layer = "input_1"
output_layer = "k2tfout_0"
estimator = Estimator(model_file, input_layer, output_layer)

cap = cv2.VideoCapture(0)

while cap.isOpened():
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	hands = detector(gray)
	for hand in hands:
		top, bottom, left, right = hand.top() - 50, hand.bottom() + 50, hand.left() - 50, hand.right() + 50

		cv2.rectangle(frame, (left, bottom), (right, top),
					  color=(0, 255, 255), thickness=2)

		# landmarks = predictor(gray, hand)
		# for i in range(NPOINTS):
		# 	cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 6, color=(255, 255, 255), thickness=-1)
		# 	# cv2.putText(frame, str(i), (landmarks.part(i).x, landmarks.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
		# for pair in PAIRS:
		# 	x0, y0 = landmarks.part(pair[0]).x, landmarks.part(pair[0]).y
		# 	x1, y1 = landmarks.part(pair[1]).x, landmarks.part(pair[1]).y
			# cv2.line(frame, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=2)

		crop = frame[top: bottom, left: right]
		if crop.size > 0:
			crop_res = cv2.resize(crop, (box_size, box_size))
			img, pad = preprocess(crop_res, box_size, stride)

			hm = estimator.predict(img)
			hm = cv2.resize(hm, (0, 0), fx=stride, fy=stride)

			points = []

			for i in range(21):
				m = hm[:, :, i]
				m_resize = cv2.resize(m, (crop.shape[1], crop.shape[0]))
				idx = np.unravel_index(np.argmax(m_resize, axis=None), m_resize.shape)
				if m_resize[idx] > THRESHOLD:
					points.append((idx[1] + left, idx[0] + top))
					cv2.circle(frame, points[i], 4, (255, 255, 255), thickness=-1)
				else:
					points.append(None)

			for pair in POSE_PAIRS:
				partA = pair[0]
				partB = pair[1]
				if points[partA] and points[partB]:
					cv2.line(frame, points[partA], points[partB], (255, 255, 255), 1, lineType=cv2.LINE_AA)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	cv2.imshow("hand detection", frame)

cap.release()
cv2.destroyAllWindows()
