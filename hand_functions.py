import cv2
import numpy as np

from hand_tracker import HandTracker

#        8    12   16  20
#        |    |    |   |
#        7    11   15  19
#    4   |    |    |   |
#    |   6    10   14  18
#    3   |    |    |   |
#    |   5----9----13--17
#    2    \   |   /    /
#     \    \  |  /    /
#      1    \ | /    /
#       \    \|/    /
#        -----0-----
connections = [
	(0, 1), (0, 5), (0, 17), (0, 9), (0, 13),
	(5, 9), (9, 13), (13, 17)
]

# Format as ((color, kp_idx), ...)
fingers = [
	((143, 30, 145), (0,  1,  2,  3,  4)),
	((158, 144, 0), (0, 5, 6, 7, 8)),
	((0, 157, 250), (0, 9, 10, 11, 12)),
	((81, 0, 176), (0, 13, 14, 15, 16)),
	((177, 106, 57), (0, 17, 18, 19, 20))
]

THICKNESS = 2
POINT_COLOR = (255, 150, 150)
CONNECTION_COLOR = (175, 175, 175)


class GestureDetector:
	def __init__(self, palm_recog_path, kp_recog_path, anchor_path):
		self.detector = HandTracker(
			palm_recog_path,
			kp_recog_path,
			anchor_path,
			box_shift=0.2,
			box_enlarge=1.3
		)
		self.finger_bend = [False] * 5

	"""
		Extract keypoints from FRAME and draw on REAL.
	"""
	def detect_hands_and_draw_skeleton(self, frame, real):
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		points, _ = self.detector(image)
		if points is not None:
			for point in points:
				x, y = point
				cv2.circle(real, (int(x), int(y)), THICKNESS * 3, POINT_COLOR, -1)
			for connection in connections:
				x0, y0 = points[connection[0]]
				x1, y1 = points[connection[1]]
				cv2.line(real, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
			for i in range(5):
				finger = fingers[i]
				color = finger[0]
				kp_idx = finger[1]
				bend = self.is_bend(i, points)

				for j in range(1, len(kp_idx) - 1):
					i1 = kp_idx[j]
					i2 = kp_idx[j + 1]
					x0, y0 = points[i1]
					x1, y1 = points[i2]
					if not self.finger_bend[i]:
						cv2.line(real, (int(x0), int(y0)), (int(x1), int(y1)), color, THICKNESS)
					else:
						cv2.line(real, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
		return points

	"""
		Returns True iff the three joint angles are all greater than 120 degrees.
	"""
	def is_bend(self, finger_idx, keypoints):
		finger = fingers[finger_idx][1]
		if finger_idx == 0:
			finger = finger[1:]
		kp = list([keypoints[i] for i in finger])
		for i in range(len(kp) - 2):
			i0 = kp[i]
			i1 = kp[i + 1]
			i2 = kp[i + 2]
			angle = self.calc_angle(i0, i1, i2)
			if angle < 120:
				self.finger_bend[finger_idx] = True
				return True
		self.finger_bend[finger_idx] = False
		return False

	"""
		Calculate the angle between ab and ac.
	"""
	def calc_angle(self, a, b, c):
		ba = a - b
		bc = c - b

		cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
		angle = np.arccos(cosine_angle)

		return np.degrees(angle)



	def detect_start_gesture(self, keypoints):
		pass
