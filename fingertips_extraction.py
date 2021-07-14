import cv2
import numpy as np

def show_image(label, frame):
	cv2.imshow(label, frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def saveFingerCrop(i, box, frame_2):
	rect = cv2.minAreaRect(box)
	width = int(rect[1][0])
	height = int(rect[1][1])

	src_pts = box.astype("float32")
	# coordinate of the points in box points after the rectangle has been
	# straightened
	dst_pts = np.array([[0, height-1],[0, 0],[width-1, 0],[width-1, height-1]], dtype="float32")

	# the perspective transformation matrix
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	# directly warp the rotated rectangle to get the straightened rectangle
	warped = cv2.warpPerspective(frame_2, M, (width, height))
	if warped.shape[0] < warped.shape[1]:
		warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

	cv2.imwrite(f"finger_{i}.jpg", warped)

def getSkinMask(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	skin_mask = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)[1]

	# frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# an = 5
	# blurSize=7
	# #it detects skin color & returns binary of it
	# skin_mask = cv2.inRange(frame_hsv, (0,58.65,50),(50,173.4,255))
	# skin_mask = cv2.medianBlur(skin_mask, blurSize)
	# a = frame.shape[0]
	# col = frame.shape[1]
	# for i in range(0, 15):
	# 	skin_mask = cv2.line(skin_mask, (0, a), (col, a), (0, 0, 0), 10)
	# 	a -= 10
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(an*2+1, an*2+1))
	# skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
	# skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

	return skin_mask


def getFingerROI(frame):
	skin_mask = getSkinMask(frame)

	contours, hierarchy = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# frame = cv2.drawContours(frame, contours, -1, (0,255,0), 5)
	c = max(contours, key=lambda cnt: cv2.contourArea(cnt))

	x,y,w,h = cv2.boundingRect(c)
	# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)

	##hardcoded value to separate fingers from palm
	h-=680
	frame_2 = frame[y:y+h,x:x+w]
	canny_img = getSkinMask(frame_2)
	contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	contours.sort(key=cv2.contourArea, reverse=True)
	contours = contours[:4]

	##contours and rotated rect are drawn acound each finger and palm
	contours_poly = [None]*len(contours)
	minRect = [None]*len(contours)
	for i, c in enumerate(contours):
		contours_poly[i] = cv2.approxPolyDP(c, 3, True)
		minRect[i] = cv2.minAreaRect(contours_poly[i])

	##reduce the finger rotated tect to the tip fingerprint region
	for i in range(len(contours)):
		if cv2.contourArea(contours[i]) > 500 :
			color = (0,255,0)
			box = cv2.boxPoints(minRect[i])
			box = np.intp(box)
			# cv2.drawContours(frame_2, [box], 0, color, 12)
			length = np.linalg.norm(box[0] - box[1])
			width = np.linalg.norm(box[0] - box[3])
			#scale = 3
			if (length > width) :
				scale = length / (1.6*width)
				box[0] =  ((scale-1)*box[1] + box[0])/scale
				box[3] =  ((scale-1)*box[2] + box[3])/scale
			if (width >= length ) :
				scale = width / (1.6*length)
				box[1] =  ((scale-1)*box[2] + box[1])/scale
				box[0] =  ((scale-1)*box[3] + box[0])/scale

			cv2.drawContours(frame_2, [box], 0, color, 12)
			saveFingerCrop(i, box, frame_2)

	show_image("roi", frame_2)

frame = cv2.imread('/Users/sreekant/Desktop/fingerprint/testing/original.jpg')
getFingerROI(frame)
