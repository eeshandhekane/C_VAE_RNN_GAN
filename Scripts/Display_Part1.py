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
print('[DEBUG] Loaded array shape : ' + str(array.shape))
index = 0

# Orig and Recn
orig = array[0]
recn = array[1]
for num_batch in range(5):
	b = np.random.randint(32)
	for num_unfold in range(3):
		u = np.random.randint(5)
		# print('[DEBUG] Shape of the original images : ' + str(orig[b][u].shape))
		# print('[DEBUG] Shape of the reconstructed images : ' + str(recn[b][u].shape))
		cv2.imwrite('temp' + str(index).zfill(5) +'.jpg', orig[b][u]*255)
		index += 1
		cv2.imwrite('temp' + str(index).zfill(5) +'.jpg', recn[b][u]*255)
		index += 1
os.system('ffmpeg -f image2 -r 1/1 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Reconstruction_Movie_Part1.mp4')
time.sleep(0.5)
os.system('rm -f temp*.jpg')
time.sleep(0.5)
