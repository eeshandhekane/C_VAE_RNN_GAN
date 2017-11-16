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


# Select randomly which sequence to select
for n in range(32):
	a_batch = array[n]
	print('[DEBUG] A batch shape : ' + str(a_batch.shape))
	print('[DEBUG] A batch entry shape : ' + str(a_batch[0].shape))
	#sys.exit()
	im_1 = a_batch[0]
	im_2 = a_batch[1]
	im_3 = a_batch[2]
	im_4 = a_batch[3]
	im_5 = a_batch[4]
	# while 1:
	# 	cv2.imshow('temp1.png', im_1)
	# 	if cv2.waitKey(5) and 0xFF == ord('q'):
	# 		break
	cv2.imwrite('temp00001.jpg', np.reshape(im_1, [64, 64])*255)
	cv2.imwrite('temp00002.jpg', np.reshape(im_2, [64, 64])*255)
	cv2.imwrite('temp00003.jpg', np.reshape(im_3, [64, 64])*255)
	cv2.imwrite('temp00004.jpg', np.reshape(im_4, [64, 64])*255)
	cv2.imwrite('temp00005.jpg', np.reshape(im_5, [64, 64])*255)
	#cv2.imwrite('temp00001.jpg', im_1)
	#cv2.imwrite('temp00001.jpg', im_2)
	#cv2.imwrite('temp00001.jpg', im_3)
	#cv2.imwrite('temp00001.jpg', im_4)
	#cv2.imwrite('temp00001.jpg', im_5)
	#os.system('ffmpeg -f image2 -r 1/5 -i image%05d.png -vcodec mpeg4 -y movie.mp4')
	os.system('ffmpeg -f image2 -r 1/1 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Movie_' + str(n) + '.mp4')
	time.sleep(0.5)
	os.system('rm -f temp00001.jpg temp00002.jpg temp00003.jpg temp00004.jpg temp00005.jpg temp00006.jpg')
	time.sleep(0.5)





