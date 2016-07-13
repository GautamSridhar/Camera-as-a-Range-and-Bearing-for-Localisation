import numpy as np
import Quadcopter
import time 


# connection with Quadcopter
quad = Quadcopter.Quadcopter(19999)
print 'Connection established with simulator'
height = quad.get_position()[2]

# load trajectory
import cPickle
fo = open('trajectory3','rb')
trajectory = cPickle.load(fo)
fo.close()

# playing back trajectory
for pos in trajectory:
	pos[2] = height #the height shouldn't change for test case
	quad.set_target_position(pos)
	time.sleep(0.1)

print 'Playback finished'

