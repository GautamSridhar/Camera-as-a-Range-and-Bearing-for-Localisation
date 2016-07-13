import vrep
import numpy as np
import cv2 
import time 

class Quadcopter():
	"""Get data from and to V-REP"""
	def __init__(self,port_number):

		vrep.simxFinish(-1)
		self.clientID = vrep.simxStart('127.0.0.1', port_number, True, True, 5000, 5)

		# start simulation
		rc = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
		
		# object handles
		res, self.quad_obj = vrep.simxGetObjectHandle(self.clientID, 'Quadricopter', vrep.simx_opmode_oneshot_wait)
		res, self.camera   = vrep.simxGetObjectHandle(self.clientID, 'Vision_sensor', vrep.simx_opmode_oneshot_wait)
		res, self.target   = vrep.simxGetObjectHandle(self.clientID, 'Quadricopter_target', vrep.simx_opmode_oneshot_wait)
		
		# Initialise data streaming from V-REP
		err, resolution, image  = vrep.simxGetVisionSensorImage(self.clientID, self.camera, 0, vrep.simx_opmode_streaming)
		_,pos                   = vrep.simxGetObjectPosition(self.clientID, self.quad_obj, -1, vrep.simx_opmode_streaming)
		_,target_pos            = vrep.simxGetObjectPosition(self.clientID, self.target, -1, vrep.simx_opmode_streaming)
		
		time.sleep(2)

		# Variables
		_,self.last_pos = vrep.simxGetObjectPosition(self.clientID, self.quad_obj, -1, vrep.simx_opmode_buffer)
		
	def get_position(self):
		'Get the quadcopters position'
		_, pos = vrep.simxGetObjectPosition(self.clientID, self.quad_obj, -1, vrep.simx_opmode_buffer)
		return pos 

	def get_odometry(self):
		'Get the del_x and del_y between current and previous position'
		cur_pos = self.get_position()
		perfect_odometry = [cur_pos[0]-self.last_pos[0], cur_pos[1]-self.last_pos[1]]

		noise_odometry   = [perfect_odometry[0] + (perfect_odometry[0]*np.random.normal(0,0.5)) ,  perfect_odometry[1] + (perfect_odometry[1]*np.random.normal(0,0.5))  ]
		
		#update last position
		self.last_pos = cur_pos
		return noise_odometry

	def get_target_position(self):
		'Get the position of the target'
		_, target_pos   = vrep.simxGetObjectPosition(self.clientID, self.target, -1, vrep.simx_opmode_buffer)
		return target_pos

	def set_target_position(self,target_position):
		'Set the target position in the simulation'
		rc = vrep.simxSetObjectPosition(self.clientID, self.target, -1, target_position, vrep.simx_opmode_oneshot)

	def get_camera_image(self):
		'Get the image from the camera'

		err, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.camera, 0, vrep.simx_opmode_buffer)
		img = np.array(image,dtype=np.uint8)
		img.resize([resolution[1],resolution[0],3])
		img = np.fliplr(img)
		temp = np.copy(img[:,:,0])
		img[:,:,0] = np.copy(img[:,:,2])
		img[:,:,2] = temp 
		return img 

