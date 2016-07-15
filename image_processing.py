import numpy as np
import cv2


# --------------------- TEMPLATE MATCHING METHOD ------------------------------

def temp_match(T,I):

	res = cv2.matchTemplate(I, T, cv2.TM_CCORR_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	w,h,_ = T.shape 
	position = [ int(max_loc[1]+(h/2.0)) , int(max_loc[0]+(w/2.0)) ]

	return position, max_val 

def find_marker_template_method(marker_list, marker_names, camera_img):
	'Finds the marker (if present) in the camera image using TM method'

	outputs = []
	for i,marker in enumerate(marker_list):
		position, max_val = temp_match(marker, camera_img)
		outputs.append([position, max_val, marker_names[i]])

	outputs.sort(key=lambda thing: thing[1])
	outputs.reverse()
	best_match = outputs[0]
	found = 0
	if best_match[1]>0.960:
		
		found = 1 

	return found, best_match[0], best_match[2]


# ------------------- CONTOUR MATCHING METHOD --------------------------------

def knn(test, train_images):
	'compare test against train_images'

	test_ravel = np.ravel(test)
	train_ravel = [np.ravel(img) for img in train_images]

	# using mean squared error metric
	min_metric = np.sum((test_ravel.astype("float") - train_ravel[0].astype("float")) ** 2)
	min_metric /= float(test.shape[0] * test.shape[1])
	min_idx = 0

	for i,img in enumerate(train_ravel):

		metric = np.sum((test_ravel.astype("float") - train_ravel[i].astype("float")) ** 2)
		metric /= float(test.shape[0] * test.shape[1])
		
		if metric<min_metric:
			min_metric = metric
			min_idx    = i 

	return min_idx, min_metric

def marker_contour(frame):
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	filter_gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edges = cv2.Canny(filter_gray, 100, 200)

	cnts,_ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]	
	
	if len(cnts)>0:
		peri = cv2.arcLength(cnts[0], True)
		approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
  
		if len(approx) == 4:
			# candidate for marker    
			x , y , w, h = cv2.boundingRect(cnts[0])
			roi = frame[y: y + h, x: x + w] 
			pos = [y+(h/2.0), x+(w/2.0)]
			return 1, roi, pos
		else:
			return 0,0,0
	else:
		return 0,0,0


def find_marker_contour_method(marker_list, marker_names, camera_img):
	'Finds the marker (if present) in the camera image using contour method'

	# locate marker
	rc, marker_detected, pos = marker_contour(camera_img)

	if rc==1 and abs(marker_detected.shape[0] - marker_detected.shape[0])<5:
		
		# Compare training images with test image
		min_idx, min_metric = knn( cv2.resize(marker_detected,(marker_list[0].shape[0],marker_list[0].shape[0])), marker_list)

		if min_metric<=30000:
			found = 1
			#print marker_names[min_idx], min_metric
			return found, pos, marker_names[min_idx]
		else:
			return 0,0,0
	else:
		return 0,0,0
def PcTem(T,I):
	w1,h1 = T.shape
	w2,h2 = I.shape
	w3 = w1+w2-1
	h3 = h1+h2-1
	Tn = np.zeros((w3,h3))
	In = np.zeros((w3,h3))
	R = np.zeros((w3,h3))
	Tn[:w1,:h1] = T
	In[:w2,:h2] = I
	dft_Tn = np.fft.fft2(Tn)
	dft_Tn = np.fft.fftshift(dft_Tn)
	dft_In = np.fft.fft2(In)
	dft_In = np.fft.fftshift(dft_In)
	Tnconj = np.conjugate(dft_Tn)
	numer = np.multiply(dft_In,Tnconj)
	denom = np.absolute(numer)
	R = np.divide(numer,denom)
	R = np.fft.ifftshift(R)
	r = np.fft.ifft2(R)
	r = np.absolute(r)
	q = r.max()
	maxloc = np.where(r == r.max())
	position = [int(maxloc[1]+(h1/2.0)) , int(maxloc[0]+(w1/2.0))]
	return maxloc, q

def find_marker_phase_method(marker_list, marker_names, camera_img):
	outputs = []
	for i,marker in enumerate(marker_list):
		position, max_val = temp_match(marker, camera_img)
		outputs.append([position, max_val, marker_names[i]])

	outputs.sort(key=lambda thing: thing[1])
	outputs.reverse()
	best_match = outputs[0]
	found = 0
	if best_match[1]>0.960:
		
		found = 1 

	return found, best_match[0], best_match[2]


