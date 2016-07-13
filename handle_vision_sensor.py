import vrep
import numpy as np
import time 
import cv2 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimage
import os 
import sys

plt.ion()

# ----- Template Matching Functions ------
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
	best_match = outputs[0]
	found = 0
	if best_match[1]>0.93:
		found = 1 

	return found, best_match[0], best_match[2]


# load the marker images 
marker_images_name = os.listdir('./Markers/small/')
marker_images = []
for name in marker_images_name:
	img = cv2.imread('./Markers/small/{}'.format(name))
	marker_images.append(img)

# VREP Code

vrep.simxFinish(-1)
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if clientID!=-1:
	print 'Connected to V-REP'

	# get vision sensor object handle 
	res, vs = vrep.simxGetObjectHandle(clientID, 'Vision_sensor', vrep.simx_opmode_oneshot_wait)

	# getting other object handles
	res, quad = vrep.simxGetObjectHandle(clientID, 'Quadricopter', vrep.simx_opmode_oneshot_wait)
	time.sleep(2)
	res, marker = vrep.simxGetObjectHandle(clientID, 'Plane1', vrep.simx_opmode_oneshot_wait)
	time.sleep(2)

	print quad, marker

	# start getting data from vision sensor
	err, resolution, image = vrep.simxGetVisionSensorImage(clientID, vs, 0, vrep.simx_opmode_streaming)
	time.sleep(2)

	
	
	while True:

		err, resolution, image = vrep.simxGetVisionSensorImage(clientID, vs, 0, vrep.simx_opmode_buffer)
		img = np.array(image,dtype=np.uint8)
		img.resize([resolution[1],resolution[0],3])
		img = np.fliplr(img)

		temp = np.copy(img[:,:,0])
		img[:,:,0] = np.copy(img[:,:,2])
		img[:,:,2] = temp 

		found, position, marker_id = find_marker(marker_images, marker_images_name,	img)
		# if found==1:
		# 	print 'Found marker {}'.format(marker_id)
		# 	img[position[0]-10:position[0]+10, position[1]-10:position[1]+10, 0] = 200
		# 	img[position[0]-10:position[0]+10, position[1]-10:position[1]+10, 1:2] = 50 
		# 	_,marker_pos = vrep.simxGetObjectPosition(clientID, marker, -1, vrep.simx_opmode_oneshot_wait)
		# 	print 'Image  detect-> x: {}  y: {}      Actual Marker: {}  {}'.format(position[1]-256, -(position[0]-256), marker_pos[0], marker_pos[1])
		# 	_,quad_pos   = vrep.simxGetObjectPosition(clientID, quad, -1, vrep.simx_opmode_oneshot_wait)
		# 	print 'Quad position: {}   {}'.format(quad_pos[0], quad_pos[1])
		#cv2.imshow('op', img)
		#cv2.waitKey(10)
		cv2.imwrite('e.png', img)
		sys.exit(0)


else:
	print 'Unable to connect to V-REP'
	vrep.simxFinish(clientID)


