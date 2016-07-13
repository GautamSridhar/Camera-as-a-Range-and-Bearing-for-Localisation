import cv2 
import numpy as np
from matplotlib import pyplot as plt
import sys 
import os 


def temp_match(T,I):

	res = cv2.matchTemplate(I, T, cv2.TM_CCORR_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	w,h,_ = T.shape 
	position = [ int(max_loc[1]+(h/2.0)) , int(max_loc[0]+(w/2.0)) ]

	return position, max_val 

def find_marker(marker_list, marker_names, camera_img):
	'Finds the marker (if present) in the camera image'

	outputs = []
	for i,marker in enumerate(marker_list):
		position, max_val = temp_match(marker, camera_img)
		outputs.append([position, max_val, marker_names[i]])

	outputs.sort(key=lambda thing: thing[1])
	outputs.reverse()
	print outputs
	best_match = outputs[0]
	found = 0
	if best_match[1]>0.98:
		
		found = 1 

	return found, best_match[0], best_match[2]



test_images_name = os.listdir('./Test_Images/')
marker_images_name = os.listdir('./Markers/small/')

print test_images_name
print marker_images_name

marker_images = []
test_images = []
for name in test_images_name:
	img = cv2.imread('./Test_Images/{}'.format(name))
	test_images.append(img)

for name in marker_images_name:
	img = cv2.imread('./Markers/small/{}'.format(name))
	marker_images.append(img)


# print 'Testing...'

found, position, marker_id = find_marker(marker_images, marker_images_name,	test_images[4])
if found==1:
	cv2.circle(test_images[4], (position[1], position[0]), 10, (0,255,255), 3)
	cv2.imshow('op', test_images[4])
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# for i in range(len(test_images)):

# 	for k in range(len(marker_images)):

# 		position, max_val = temp_match(marker_images[k], test_images[i])
# 		print 'Marker:{}   test: {}   val: {}'.format(marker_images_name[k], test_images_name[i], max_val)
