import numpy as np
import cv2
import dlib
import time


THICKNESS = 1
POINT_COLOR = (255, 150, 150)
CONNECTION_COLOR = (0, 200, 255)


class FaceDetector:
	def __init__(self, predictor_path):
		# Save up to 3 faces to freeze. Format as (crop, mask, velocity)
		self.freeze_faces = []

		# Save up to 2 history central points to determine velocity. Format as (x, y, time)
		self.history_central = []

		# facial detector & shape_predictor from dlib
		self.face_detector = dlib.get_frontal_face_detector()
		self.face_predictor = dlib.shape_predictor(predictor_path)

	"""
		Use LANDMARKS output by self.face_predictor to find outline for face and mouth
		and draw outlines on REAL and FRAME
	"""
	def find_face_and_mouth(self, frame, real, landmarks):
		# get border of face
		face_outline = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17)]
		face_outline.extend([(landmarks.part(i).x, landmarks.part(i).y) for i in range(26, 16, -1)])
		face_outline = np.array(face_outline)

		# get border of mouth
		mouth_outline = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)])

		# save central point to history
		self.history_central.append((landmarks.part(30).x, landmarks.part(30).y, time.time()))
		if len(self.history_central) > 2:
			self.history_central.pop(0)

		# draw outline for face and mouth
		cv2.drawContours(frame, [face_outline, mouth_outline], -1, CONNECTION_COLOR, THICKNESS)
		cv2.drawContours(real, [face_outline, mouth_outline], -1, CONNECTION_COLOR, THICKNESS)

		# draw points for 60 keypoints
		for i in range(60):
			x = landmarks.part(i).x
			y = landmarks.part(i).y
			cv2.circle(frame, (x, y), THICKNESS * 2, POINT_COLOR, -1)
			cv2.circle(real, (x, y), THICKNESS * 2, POINT_COLOR, -1)

		return [face_outline, mouth_outline]

	"""
		Generate mask that highlights area in FACE_OUTLINE.
	"""
	def gen_mask(self, frame, face_outline):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		mask = np.zeros_like(gray)
		cv2.drawContours(mask, [face_outline], -1, 255, -1)
		return mask

	"""
		Find 68 face landmarks using detector and predictor using grayscale image of frame.
		Return face-landmarks pairs: [[f1, l1], [f2, l2], ...]
	"""
	def find_landmarks(self, gray):
		res = []
		faces = self.face_detector(gray)
		for face in faces:
			landmarks = self.face_predictor(gray, face)
			res.append([face, landmarks])
		return res

	"""
		Combines find_landmarks and find_face_and_mouth.
		Returns face-mouth pairs: [[f1, m1], [f2, m2], ...]
	"""
	def face_pipeline(self, frame, real):
		res = []
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		face_landmarks = self.find_landmarks(gray)
		for face, landmarks in face_landmarks:
			face_outline, mouth_outline = self.find_face_and_mouth(frame, real, landmarks)
			res.append([face_outline, mouth_outline])
		return res

	"""
		Overlay faces stored in self.freeze_faces on real 
		and adjust location for stored faces by velocity and time.
	"""
	def overlay_face(self, real, dt):
		for i in range(len(self.freeze_faces)):
			crop, mask, v = self.freeze_faces[i]

			# overlay faces on real
			real[mask == 255] = crop[mask == 255]

			# calculate displacement
			dx = v[0] * dt
			dy = v[1] * dt

			# transform matrix to feed into warpAffine: move (dx, dy)
			T = np.float32([[1, 0, np.ceil(dx)], [0, 1, np.ceil(dy)]])
			rows, cols = self.freeze_faces[i][1].shape

			# shift both mask and crop
			self.freeze_faces[i][1] = cv2.warpAffine(self.freeze_faces[i][1], T, dsize=(cols, rows))
			self.freeze_faces[i][0] = cv2.warpAffine(self.freeze_faces[i][0], T, dsize=(cols, rows))
