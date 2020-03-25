import cv2

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \  |   |  /
#     \    \ |   / /
#      1    \ \ / /
#       \    \ / /
#        ------0-
connections = [
	(0, 1), (1, 2), (2, 3), (3, 4),
	(5, 6), (6, 7), (7, 8),
	(9, 10), (10, 11), (11, 12),
	(13, 14), (14, 15), (15, 16),
	(17, 18), (18, 19), (19, 20),
	(0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
	(0, 9), (0, 13)
]

THICKNESS = 1
POINT_COLOR = (255, 255, 255)
CONNECTION_COLOR = (255, 255, 255)


def detect_hands_and_draw_skeleton(frame, real, detector):
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	points, _ = detector(image)
	if points is not None:
		for point in points:
			x, y = point
			cv2.circle(real, (int(x), int(y)), THICKNESS * 3, POINT_COLOR, 1)
		for connection in connections:
			x0, y0 = points[connection[0]]
			x1, y1 = points[connection[1]]
			cv2.line(real, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
