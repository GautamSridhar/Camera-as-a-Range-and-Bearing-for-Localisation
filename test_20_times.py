import numpy as np
import os 
import sys 
import time

def main():

	for i in range(20):

		print '\n  Starting Iteration {}'.format(i)
		os.system('python playback_trajectory.py &')
		time.sleep(3)
		os.system('python localization.py')
		time.sleep(10)
		print '\n  Iteration {} ended'.format(i)


if __name__=='__main__':
	main()