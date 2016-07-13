import numpy as np
import cv2 
import matplotlib.pyplot as plt
from Quadcopter import Quadcopter 
import time 
import image_processing
import os 
import sys 
import math


plt.style.use('ggplot')

def predict_state(x,P,u):
	'Predict the next state given the previous state, covariance and the odometer values'

	x_priori = x + np.array([ [u[0]] , [u[1]] ])
	F_x      = np.eye(2)
	F_u      = np.eye(2)
	k        = 0.9
	Q        = np.array([ [ k*abs(u[0]) , 0] , [ 0 , k*abs(u[1]) ]  ])
	P_priori = np.dot(F_x, np.dot(P, F_x.T)) + Q

	return x_priori, P_priori 


def correct_state(x,P,camera_img, marker_images, marker_images_name, scale_factor):
	'Correct the state given the camera image'

	#found, position, marker_id = image_processing.find_marker_template_method(marker_images, marker_images_name, camera_img)
	found, position, marker_id = image_processing.find_marker_contour_method(marker_images, marker_images_name, camera_img)
	marker_positions = [ [-0.4250,-0.25], [1.9750,-0.2750], [1.9750,2.00],[-0.5250,2.00], [-3.00,1.9750], [-3.00,-0.25], [-3.00,-2.3250] ]

	if found==1:
		# convert to cortesian coord
		#z = np.array([ [( (position[1]*2)-256)*3.330579756*0.001] , [-( (position[0]*2)-256)*3.330579756*0.001]  ])
		z = np.array([ [( (position[1])-128)*scale_factor] , [-( (position[0])-128)*scale_factor]  ])

		# find actual_position
		actual_marker_position = marker_positions[marker_images_name.index(marker_id)]
		z_cap = np.array([ [actual_marker_position[0] - x[0,0]] , [actual_marker_position[1]- x[1,0]] ])

		# innovation
		v   = z - z_cap
		H_x = np.eye(2)*-1
		R   = np.eye(2)*0.01
		S   = np.dot(H_x, np.dot(P,H_x.T)) + R  
		
		# Kalman Gain
		K   = np.dot(P, np.dot(H_x.T, np.linalg.inv(S)))

		# Correction update step
		x_posterori = x + np.dot(K,v)
		P_posterori = np.dot( np.eye(2)-np.dot(K,H_x)  , P)

		return x_posterori, P_posterori
	else:
		return x,P

def main():
	'Main function'

	# KF variables
	x = np.zeros((2,1))
	P = np.eye(2)*0.01

	# VREP quadcopter
	quad = Quadcopter(19997)
	last_pos = quad.get_position()
	x[0,0] = last_pos[0]
	x[1,0] = last_pos[1]

	#load the calibration data
	import cPickle
	fo = open('calibration.p','rb')
	calib = cPickle.load(fo)
	fo.close()
	ms = calib(last_pos[2])

	#calculate scaling factor
	scale_factor = 0.2837/ms

	# Load scaled markers (according to height calibration)
	marker_images = []
	marker_images_name = os.listdir('./Markers/originals/')
	for name in marker_images_name:
		img = cv2.imread('./Markers/originals/{}'.format(name))
		marker_images.append(cv2.resize(img,(int(ms),int(ms))))
	print marker_images_name

	# Matrices to store the positions - estimated and actual
	nIter = 1000
	actual_trajectory = np.zeros((nIter,2))
	estimated_trajectory = np.zeros((nIter,2))

	# matplotlib  plotting 
	fig, ax = plt.subplots()
	ax.set_xlim(-5,5)
	ax.set_ylim(-5,5)
	ax.set_title('Localization')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	line1, = ax.plot(x[0,0], x[1,0],'go')
	ax.hold(True)
	line2, = ax.plot(x[0,0],x[1,0],'ro')
	ax.hold(False)


	for i in range(nIter):

		# --------- Predict State ------------
		u   = quad.get_odometry()
		x,P = predict_state(x, P, u)

		# # --------  Correct State ------------
		if i%10==0:
			camera_img = quad.get_camera_image()
			x,P = correct_state(x, P, camera_img, marker_images, marker_images_name, scale_factor)

		# Save data
		actual_trajectory[i,:] = x.T
		estimated_trajectory[i,:] = np.array(quad.get_position()[0:2]) 

		# plot state
		# estimated state
		line1.set_xdata(x[0,0])
		line1.set_ydata(x[1,0])

		# actual state
		actual_pos = quad.get_position()
		line2.set_xdata(actual_pos[0])
		line2.set_ydata(actual_pos[1])

		plt.pause(0.001)


	print 'Finished Localization'

	# plot trajectory and error
	plt.close()
	# plt.subplot(121)
	# plt.plot(actual_trajectory[:,0],actual_trajectory[:,1],'r')
	# plt.hold(True)
	# plt.plot(estimated_trajectory[:,0],estimated_trajectory[:,1],'g')
	# plt.axis([-5,5,-5,5])
	# plt.hold(False)
	# plt.subplot(122)
	error = np.square(actual_trajectory[:,0:2] - estimated_trajectory[:,0:2])
	error = np.sum(error,axis=1)
	error = np.sqrt(error)
	# plt.plot(error)
	# plt.show()

	# save error to file
	np.save('error_{}'.format(str(time.clock())), error)




if __name__=='__main__':
	main()

