# Dependencies
import cv2
import numpy as np
import argparse
import os, sys, re
import time


# Parse the npy name
parser = argparse.ArgumentParser()
parser.add_argument('--npy_name', type = str, required = True)
#parser.add_argument('--num_im', type = int, required = True)
args = parser.parse_args()
npy_name = str(args.npy_name)
#num_im = str(args.num_im)


# Load the npy
array = np.load(npy_name)
print('[DEBUG] Loaded array shape : ' + str(array.shape)) # [32, 1, 32, 5, 64, 64, 1]
#of_interest = np.random.randint(32)
array_of_interest = array[0][0] # To remove the extra dimension
print('[DEBUG] Loaded array of interest shape : ' + str(array_of_interest.shape)) # [32, 5, 64, 64, 1]


# Generate 32 movies! DEUS VULT
for n in range(32):
	index = 0
	movie = array_of_interest[n]
	print('[DEBUG] Movie shape : ' + str(movie.shape))
	for i in range(5):
		cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', movie[index]*255)
		index = index + 1
	os.system('ffmpeg -f image2 -r 1/1 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'C_RNN_GAN_Generated_Video' + str(n) + '.mp4')
	time.sleep(0.5)
	os.system('rm -f temp*.jpg')
	time.sleep(0.5)
