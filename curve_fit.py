import numpy as np
import matplotlib.pyplot as plt 


calibration_data = np.array([  [0.99,64],  
	                           [1.24,51.0098],
	                           [1.49,42.011],
	                           [1.74,34.0],
	                           [1.99,31.0],
	                           [2.24,28],
	                           [2.49,24.009] ])

z = np.polyfit( calibration_data[:,0], calibration_data[:,1], 4)
p = np.poly1d(z)

plt.ion()
#plt.style.use('ggplot')
plt.plot(calibration_data[:,1], calibration_data[:,0], 'ro')
plt.hold(True)
plt.plot(  p(np.arange(0.75,2.75,0.1)) , np.arange(0.75,2.75,0.1), 'g' )
plt.hold(False)
plt.xlabel('Marker Size (pixels)')
plt.ylabel('Height of Drone (m)')
plt.title('Regression plot')