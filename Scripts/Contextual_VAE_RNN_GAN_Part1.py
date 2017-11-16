####################################################################################################
############################################BASIC###################################################
####################################################################################################


# Dependencies
import tensorflow as tf
import numpy as np
import cv2
import os, sys, re
import random
import argparse
import scipy.misc


# Parameters
# Basic Parameters
IMG_SIZE = 64
BATCH_SIZE = 32
# Encoder Parameters
ENC_INPUT_CHANNEL = 1
ENC_K1 = 16
ENC_K2 = 32
ENC_K3 = 64
# Latent Parameters
HIDDEN_DIM = int(IMG_SIZE/(2*2*2*2*2)) # Needs to be computed properly
LATENT_DIM = 16
# Decoder Parameters
DEC_K1 = ENC_K3
DEC_K2 = ENC_K2
DEC_K3 = ENC_K1
DEC_OUTPUT_CHANNEL = ENC_INPUT_CHANNEL
# RNN Parameters
CELL_SIZE = LATENT_DIM
STACK_SIZE = 2
UNFOLD_SIZE = 6
# Auxiliary Parameters
LABEL_SIZE = 2 # x, y of patch for now.. Can vary with the question
# Iterations
ITR = 100000 + 1
# Training Parameters
lr_GradientDescentOptimizer = 0.00001
lr_AdamOptimizer = 0.001


# Class dataLoader
class dataLoader :
	"""
	The data loader class
	"""
	# Define the constructor
	def __init__(self, tr_prop = 0.60, val_prop = 0.20, te_prop = 0.20):
		# Information message
		print('[INFO] Creating Data Loader Instance ... ')
		self.data_dir = 'Dataset/'
		self.dir_list = os.listdir(self.data_dir)
		self.train_list = []
		self.validation_list = []
		self.test_list = []
		# Randomly distribute the datasets as train, validation and test
		for a_dir in self.dir_list :
			r = np.random.random()
			if r <= tr_prop :
				self.train_list.append(a_dir)
			elif r <= tr_prop + val_prop:
				self.validation_list.append(a_dir)
			else :
				self.test_list.append(a_dir)
		# Very rare error!!
		if len(self.train_list) == 0 or len(self.validation_list) == 0 or len(self.test_list) == 0:
			print('[ERROR] One of the data splits is empty!!')
			sys.exit()
		print('[INFO] Data Loader Initiated ... ')


	# Define a function to get the next batch
	def GetNextBatch(self, batch_size = BATCH_SIZE): 
		# Define batches
		batch_im = []
		# Select the directories from which to get sequences
		for a_batch in range(batch_size):
			batch_im.append([])
			a_dir = random.choice(self.train_list)
			# print('[DEBUG] Batch directory list : ' + str(a_dir))
			# open the dir
			this_dir = self.data_dir + a_dir
			# length from first 15 or so, so that we rarely hit the empty sequence
			first = np.random.randint(1, 10)
			diff = 1
			# Add videos
			for _ in range(6):
				next_im = cv2.imread(this_dir + '/' + str(first) + '.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
				next_im = next_im/255
				next_im = np.reshape(next_im, [IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
				batch_im[-1].append(next_im)
				first = first + diff
		batch_im_np = np.array(batch_im)
		return batch_im_np


# Generate a data loader instance
data_loader = dataLoader()
next_batch = data_loader.GetNextBatch()
np.save('Example_Batch', next_batch)
print('[DEBUG] Next batch has a peculiar and almost always a constant value of : ' + str(np.linalg.norm(next_batch)))
print('[DEBUG] Example Batch Shape : ' + str(next_batch.shape))
# print('[DEBUG] Exiting ... ')
# sys.exit()


####################################################################################################
###########################CLASSES AND FUNCTIONS OF PIX2PIX AE######################################
####################################################################################################


# Class batchNorm
class batchNorm(object):
	"""
	Callable object for batch normalization
	"""
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batchNorm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x, decay = self.momentum, updates_collections = None, epsilon = self.epsilon, scale = True, is_training = train, scope = self.name)


# Define Leaky ReLU
def LeakyReLU(x, leak=0.2):
	return tf.maximum(x, leak*x)


# Define a function to pull new variable with specs or return the one that already exists for W
def GetConvolutionalFilter(name, shape):
	return tf.get_variable(name = name + '_filter', shape = shape, initializer = tf.contrib.layers.xavier_initializer(uniform = True, seed = None, dtype = tf.float32), reuse = True)


# Define a function to pull new variable with specs or return the one that already exists for B
def GetConvolutionalBias(name, shape):
	return tf.get_variable(name = name + '_bias', shape = shape, initializer = tf.constant_initializer(0.0), reuse = True)


# Define a function to perform conv2d operations
def ConvolutionalLayer(bottom, name, shape, strides = [1,2,2,1], padding = 'SAME', bn = True, nonlin = True, is_train = True):
	'''
	bn_name will be used to use different bn params but shared weight params
	bn to regulate application of batchNorm
	nonlin to regulate application of non linearity
	'''
	# with tf.device('/cpu:0'):
	filt = GetConvolutionalFilter(name, shape)
	conv = tf.nn.conv2d(bottom, filt, strides, padding = padding)
	if bn:
		bn = batchNorm(name = 'bn_' + name)
		bias = bn(conv, train = is_train)
	else:
		# with tf.device('/cpu:0'):
		conv_bias = GetConvolutionalBias(name, [shape[-1]])
		bias = tf.nn.bias_add(conv, conv_bias)
	if nonlin:
		relu = LeakyReLU(bias)
	else:
		relu = bias
	return relu


# Define a function to perform conv2d operations
def UpconvolutionalLayer(bottom, name, shape, output_shape, strides = [1,2,2,1], padding = 'SAME',bn = True, nonlin = True, is_train = True):
	'''
	bn to regulate application of batchNorm
	nonlin to regulate application of non linearity
	'''
	# with tf.device('/cpu:0'):
	filt = GetConvolutionalFilter(name, shape)
	conv = tf.nn.conv2d_transpose(bottom, filt, output_shape, [1, 2, 2, 1], padding = padding)
	if bn:
		bn = batchNorm(name = 'bn_' + name)
		bias = bn(conv, train = is_train)
	else:
		# with tf.device('/cpu:0'):
		conv_bias = GetConvolutionalBias(name, [output_shape[-1]])
		bias = tf.nn.bias_add(conv,conv_bias)
	if nonlin:
		relu = LeakyReLU(bias)
	else:
		relu = bias
	return relu


# Define a Pix2Pix Encoder
def EncoderPix2Pix(x):
	with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):
		ENC_Y1 = ConvolutionalLayer(x, 'conv_1', [4, 4, 1, 32], strides = [1, 2, 2, 1], is_train = True) # [BATCH_SIZE, IMG_SIZE/2, IMG_SIZE/2, 32]
		ENC_Y2 = ConvolutionalLayer(ENC_Y1, 'conv_2', [4, 4, 32, 64], strides = [1, 2, 2, 1], is_train = True) # [BATCH_SIZE, IMG_SIZE/4, IMG_SIZE/4, 64]
		ENC_Y3 = ConvolutionalLayer(ENC_Y2, 'conv_3', [4, 4, 64, 128], strides = [1, 2, 2, 1], is_train = True) # [BATCH_SIZE, IMG_SIZE/8, IMG_SIZE/8, 128]
		ENC_Y4 = ConvolutionalLayer(ENC_Y3, 'conv_4', [4, 4, 128, 128], strides = [1, 2, 2, 1], is_train = True, nonlin = False) # [BATCH_SIZE, IMG_SIZE/16, IMG_SIZE/16, 128]
		# ENC_Y5 = tf.tanh(ENC_Y4)
	return ENC_Y4


# Define a Pix2Pix Decoder w/ Upconvolutions through conv2d_transpose
def DecoderPix2Pix(x):
	# In uconv filter last two indices have swapped meaning in comparison to conv
	# conv2d -> [filter_height, filter_width, in_channels, out_channels]
	# conv2d_transpose (used in uconv) -> [filter_height, filter_width, output_channels, input_channels]
	with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
		DEC_Y1 = UpconvolutionalLayer(x, 'uconv_1', [4, 4, 128, 128], [BATCH_SIZE*UNFOLD_SIZE, int(IMG_SIZE/8), int(IMG_SIZE/8), 128], is_train = True)
		DEC_Y2 = UpconvolutionalLayer(DEC_Y1, 'uconv_2', [4, 4, 64, 128], [BATCH_SIZE*UNFOLD_SIZE, int(IMG_SIZE/4), int(IMG_SIZE/4), 64], is_train = True)
		DEC_Y3 = UpconvolutionalLayer(DEC_Y2, 'uconv_3', [4, 4, 32, 64], [BATCH_SIZE*UNFOLD_SIZE, int(IMG_SIZE/2), int(IMG_SIZE/2), 32], is_train = True)
		DEC_Y4 = UpconvolutionalLayer(DEC_Y3, 'uconv_4', [4, 4, 32, 32], [BATCH_SIZE*UNFOLD_SIZE, int(IMG_SIZE), int(IMG_SIZE), 32], is_train = True)
		DEC_Y5 = ConvolutionalLayer(DEC_Y4, 'conv_5', [4, 4, 32, 1], strides=[1, 1, 1, 1], nonlin = False, bn = False, is_train = True)     
		DEC_Y6 = tf.sigmoid(DEC_Y5)
	return DEC_Y6


####################################################################################################
###################DEFINE VARIABLES, PLACEHOLDERS, LOSS AND OPTIMIZERS##############################
####################################################################################################


# Define placeholders and variables
X_im = tf.placeholder(tf.float32, [BATCH_SIZE, UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
X_im_reshaped = tf.reshape(X_im, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
H_enc = EncoderPix2Pix(X_im_reshaped)
print H_enc.shape
X_rec_reshaped = DecoderPix2Pix(H_enc)
print X_rec_reshaped.shape
X_rec = tf.reshape(X_rec_reshaped, [BATCH_SIZE, UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
X = tf.reshape(X_im_reshaped, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE*IMG_SIZE*ENC_INPUT_CHANNEL])
X_hat = tf.reshape(X_rec_reshaped, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE*IMG_SIZE*ENC_INPUT_CHANNEL])
# Linearize inputs and outputs
# LLT = tf.reduce_sum(X*tf.log(X_hat + 1e-9) + (1 - X)*tf.log(1 - X_hat + 1e-9), axis = 1)
# VLB = tf.reduce_mean(LLT)
# trainin_step = tf.train.AdamOptimizer().minimize(-VLB)
L2_loss = tf.reduce_sum((X_rec_reshaped-X_im_reshaped)*(X_rec_reshaped-X_im_reshaped))
trainin_step = tf.train.AdamOptimizer().minimize(L2_loss)


####################################################################################################
##################################Saver and Initializer#############################################
####################################################################################################


# Define session and initializer
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


####################################################################################################
###################################TRAINING ITERATIONS##############################################
####################################################################################################


for iteration in range(ITR) : 
	if iteration%500 == 0 :
		print('[TRAINING] Iteration : ' + str(iteration))
		training_batch_display = data_loader.GetNextBatch()
		L2_loss_ = sess.run(L2_loss, feed_dict = { X_im : training_batch_display })
		print L2_loss_
	if iteration%5000 == 0 :
		print('[TRAINING] Iteration : ' + str(iteration))
		print('[TRAINING] Saving Partial Model Reconstruction ... ')
		data_array = []
		training_batch_record = data_loader.GetNextBatch()
		[X_im_, X_rec_] = sess.run([X_im, X_rec], feed_dict = { X_im : training_batch_record })
		data_array.append(X_im_)
		data_array.append(X_rec_)
		data_array_np = np.array(data_array)
		np.save('Pix2Pix_VAE_Reconstruction_Iteration_' + str(iteration), data_array_np)


####################################################################################################
#############################THE VARIABLES ARE NOT REUSED###########################################
####################################################################################################