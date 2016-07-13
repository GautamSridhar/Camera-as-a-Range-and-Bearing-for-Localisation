import numpy as np
import Quadcopter
import time 


# Connection with quadcopter
quad = Quadcopter.Quadcopter(19999)

print 'Quadcopter connection established'

_ = raw_input('Press any key to begin recording trajectory...')

trajectory = []

for i in range(500):

	position = quad.get_target_position()
	trajectory.append(position)
	time.sleep(0.1)

print '-----Stopped Recording'

import cPickle
fo = open('trajectory3','wb')
cPickle.dump(trajectory, fo)
fo.close()









