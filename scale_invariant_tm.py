import numpy as np
import cv2
import matplotlib.pyplot as plt 
import image_processing
import os 

#load the calibration data
import cPickle
fo = open('calibration.p','rb')
calib = cPickle.load(fo)
fo.close()

#find marker size (in pixels) from height
height = 1.50
ms = calib(height)

# load the markers 
marker_images = []
marker_images_name = os.listdir('./Markers/originals/')
for name in marker_images_name:
	img = cv2.imread('./Markers/originals/{}'.format(name))
	marker_images.append(img)

# resize markers
marker_scaled = []
for m in marker_images:
	marker_scaled.append(cv2.resize(m,(int(ms),int(ms))))


# run template matching 
camera_img = cv2.imread('./Test_Images/b.png')
found, position, marker_id = image_processing.find_marker_template_method(marker_scaled, marker_images_name, camera_img)



