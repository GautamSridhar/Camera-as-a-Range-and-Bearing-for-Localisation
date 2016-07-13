import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time 

def contour_match(frame):
	
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


def knn(test, train_images):
	'compare test against train_images'

	test_ravel = np.ravel(test)
	train_ravel = [np.ravel(img) for img in train_images]

	# using mean squared error metric
	min_metric = np.sum((test_ravel.astype("float") - train_ravel[0].astype("float")) ** 2)
	min_metric /= float(test.shape[0] * test.shape[1])
	min_idx = 0
	
	#using ssd metric
	#min_metric = np.linalg.norm(test_ravel-train_ravel[0])
	#min_idx    = 0

	for i,img in enumerate(train_ravel):

		#metric = np.linalg.norm(test_ravel-img)
		metric = np.sum((test_ravel.astype("float") - train_ravel[i].astype("float")) ** 2)
		metric /= float(test.shape[0] * test.shape[1])

		print metric
		if metric<min_metric:
			min_metric = metric
			min_idx    = i 

	return min_idx, min_metric

def main():
	# main
	camera_img = cv2.imread('./Test_Images/g.png')
	rc, marker_detected, pos = contour_match(camera_img)


	#load the calibration data
	import cPickle
	fo = open('calibration.p','rb')
	calib = cPickle.load(fo)
	fo.close()
	height = 1.5
	ms = calib(height) #marker size


	# Load scaled markers (according to height calibration)
	marker_images = []
	marker_images_name = os.listdir('./Markers/originals/')
	for name in marker_images_name:
		img = cv2.imread('./Markers/originals/{}'.format(name))
		marker_images.append(cv2.resize(img,(int(ms),int(ms))))
	print marker_images_name


	# Compare training images with test image
	min_idx, min_metric = knn(cv2.resize(marker_detected,(int(ms),int(ms))), marker_images)
	print min_metric, marker_images_name[min_idx]



if __name__ == '__main__':
	main()
	

