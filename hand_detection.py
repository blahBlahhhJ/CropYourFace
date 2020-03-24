import cv2
import numpy as np

proto_file = "model/pose_deploy.prototxt"
weights_file = "model/pose_iter_102000.caffemodel"

NPOINTS = 22
THRESHOLD = 0.2
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
			  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

inHeight = 184
aspect_ratio = 1280 / 720
inWidth = int(((aspect_ratio * inHeight) * 8) // 8)

cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

while cap.isOpened():
	ret, frame = cap.read()
	inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
									(0, 0, 0), swapRB=False, crop=False)
	net.setInput(inpBlob)
	output = net.forward()

	points = []
	for i in range(NPOINTS):
		prob_map = output[0, i, :, :]
		prob_map = cv2.resize(prob_map, (1280, 720))
		min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

		if prob > THRESHOLD:
			cv2.circle(frame, (int(point[0]), int(point[1])),
					   radius=6, color=(0, 255, 255), thickness=-1,
					   lineType=cv2.FILLED)
			points.append((int(point[0]), int(point[1])))
		else:
			points.append(None)

	for pair in POSE_PAIRS:
		partA = pair[0]
		partB = pair[1]

		if points[partA] and points[partB]:
			cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
			cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
			cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	cv2.imshow("hand detection", frame)

cap.release()
cv2.destroyAllWindows()
