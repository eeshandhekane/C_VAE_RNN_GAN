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
CELL_SIZE = 4*4*128
STACK_SIZE = 2
UNFOLD_SIZE = 5
# Auxiliary Parameters
LABEL_SIZE = 2 # x, y of patch for now.. Can vary with the question
# Iterations
VAE_ITR = 40000 + 1
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
#########################PLACEHOLDERS AND VARIABLES DEFINITIONS#####################################
####################################################################################################


# Definitions of placeholders and variables
# Placeholdersi
X_im = tf.placeholder(tf.float32, [BATCH_SIZE, UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
X_im_reshaped = tf.reshape(X_im, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
ENC_trial_ph = tf.placeholder(tf.float32, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
DEC_trial_ph = tf.placeholder(tf.float32, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE/16, IMG_SIZE/16, 128])
# Encoder Variables
ENC_W1 = tf.Variable(tf.truncated_normal([4, 4, 1, 32], stddev = 0.1), name = 'ENC_W1')
ENC_B1 = tf.Variable(tf.zeros([32]), name = 'ENC_B1')
ENC_W2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.1), name = 'ENC_W2')
ENC_B2 = tf.Variable(tf.zeros([64]), name = 'ENC_B2')
ENC_W3 = tf.Variable(tf.truncated_normal([4, 4, 64, 128], stddev = 0.1), name = 'ENC_W3')
ENC_B3 = tf.Variable(tf.zeros([128]), name = 'ENC_B3')
ENC_W4 = tf.Variable(tf.truncated_normal([4, 4, 128, 128], stddev = 0.1), name = 'ENC_W4')
ENC_B4 = tf.Variable(tf.zeros([128]), name = 'ENC_B4')
# Decoder Variables
DEC_W4 = tf.Variable(tf.truncated_normal([4, 4, 128, 128], stddev = 0.1), name = 'DEC_W4')
DEC_B4 = tf.Variable(tf.zeros([128]), name = 'DEC_B4')
DEC_W3 = tf.Variable(tf.truncated_normal([4, 4, 64, 128], stddev = 0.1), name = 'DEC_W3')
DEC_B3 = tf.Variable(tf.zeros([64]), name = 'DEC_B3')
DEC_W2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.1), name = 'DEC_W2')
DEC_B2 = tf.Variable(tf.zeros([32]), name = 'DEC_B2')
DEC_W1 = tf.Variable(tf.truncated_normal([4, 4, 1, 32], stddev = 0.1), name = 'DEC_W1')
DEC_B1 = tf.Variable(tf.zeros([1]), name = 'DEC_B1')
# Generator Variables
GEN_gru_cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
GEN_multi_gru_cell = tf.contrib.rnn.MultiRNNCell([GEN_gru_cell]*STACK_SIZE, state_is_tuple = False)
# Discriminator Variables
DIS_gru_cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
DIS_multi_gru_cell = tf.contrib.rnn.MultiRNNCell([DIS_gru_cell]*STACK_SIZE, state_is_tuple = False)
DIS_W_dec = tf.Variable(tf.truncated_normal([CELL_SIZE, 1], stddev = 0.1), name = 'DIS_W_dec')
DIS_B_dec = tf.Variable(tf.zeros([1]), name = 'DIS_B_dec')



# Definition of Encoder Forward Pass
def EncoderPix2Pix(x, reuse = None):
	ENC_Y1 = tf.add(tf.nn.conv2d(x, ENC_W1, strides = [1, 2, 2, 1], padding = 'SAME'), ENC_B1) # [BATCH_SIZE, IMG_SIZE/2, IMG_SIZE/2, 32]
	ENC_Y2 = tf.layers.batch_normalization(ENC_Y1, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'ENC_BN1')
	ENC_Y3 = tf.nn.relu(ENC_Y2)
	ENC_Y4 = tf.add(tf.nn.conv2d(ENC_Y3, ENC_W2, strides = [1, 2, 2, 1], padding = 'SAME'), ENC_B2) # [BATCH_SIZE, IMG_SIZE/4, IMG_SIZE/4, 64]
	ENC_Y5 = tf.layers.batch_normalization(ENC_Y4, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'ENC_BN2')
	ENC_Y6 = tf.nn.relu(ENC_Y5)
	ENC_Y7 = tf.add(tf.nn.conv2d(ENC_Y6, ENC_W3, strides = [1, 2, 2, 1], padding = 'SAME'), ENC_B3) # [BATCH_SIZE, IMG_SIZE/8, IMG_SIZE/8, 128]
	ENC_Y8 = tf.layers.batch_normalization(ENC_Y7, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'ENC_BN3')
	ENC_Y9 = tf.nn.relu(ENC_Y8)
	ENC_Y10 = tf.add(tf.nn.conv2d(ENC_Y9, ENC_W4, strides = [1, 2, 2, 1], padding = 'SAME'), ENC_B4) # [BATCH_SIZE, IMG_SIZE/16, IMG_SIZE/16, 128]
	ENC_Y11 = tf.tanh(ENC_Y10)
	return ENC_Y11
Encoder_Pass_Initializer = EncoderPix2Pix(ENC_trial_ph)
print('[DEBUG] Encoder Output Shape : ' + str(Encoder_Pass_Initializer.shape))


# Definition of Decoder Forward Pass
def DecoderPix2Pix(x, reuse = None):
	DEC_Y1 = tf.add(tf.nn.conv2d_transpose(x, DEC_W4, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE/8, IMG_SIZE/8, 128], [1, 2, 2, 1]), DEC_B4)
	DEC_Y2 = tf.layers.batch_normalization(DEC_Y1, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'DEC_BN1')
	DEC_Y3 = tf.nn.relu(DEC_Y2)
	DEC_Y4 = tf.add(tf.nn.conv2d_transpose(DEC_Y3, DEC_W3, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE/4, IMG_SIZE/4, 64], [1, 2, 2, 1]), DEC_B3)
	DEC_Y5 = tf.layers.batch_normalization(DEC_Y4, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'DEC_BN2')
	DEC_Y6 = tf.nn.relu(DEC_Y5)
	DEC_Y7 = tf.add(tf.nn.conv2d_transpose(DEC_Y6, DEC_W2, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE/2, IMG_SIZE/2, 32], [1, 2, 2, 1]), DEC_B2)
	DEC_Y8 = tf.layers.batch_normalization(DEC_Y7, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'DEC_BN3')
	DEC_Y9 = tf.nn.relu(DEC_Y8)
	DEC_Y10 = tf.add(tf.nn.conv2d_transpose(DEC_Y9, DEC_W1, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, 1], [1, 2, 2, 1]), DEC_B1)
	DEC_Y11 = tf.sigmoid(DEC_Y10)
	return DEC_Y11
Decoder_Pass_Initializer = DecoderPix2Pix(DEC_trial_ph)
print('[DEBUG] Decoder Output Shape : ' + str(Decoder_Pass_Initializer.shape))


# Definition of Generator Cell
def Generator(x, GEN_h_in):
	GEN_H_r, GEN_H = tf.nn.dynamic_rnn(GEN_multi_gru_cell, x, initial_state = GEN_h_in, scope = 'GEN')
	GEN_H_r_norm = tf.tanh(GEN_H_r)
	return GEN_H_r_norm, GEN_H


# Definition of Discriminator Cell
def Discriminator(x, DIS_h_in):
	DIS_H_r, DIS_H = tf.nn.dynamic_rnn(DIS_multi_gru_cell, x, initial_state = DIS_h_in, scope = 'DIS')
	return DIS_H_r, DIS_H


# Define a function to crop out the last time stamp output from each batch
def GetLastRNNOutput(output_data): # [BATCH_SIZE, None, CELL_SIZE] shaped with dynamic None
	output_data_shape = tf.shape(output_data)
	all_batches = tf.range(output_data_shape[0])
	last_of_all_batches = tf.fill([output_data_shape[0]], output_data_shape[1]-1)
	indices_list = tf.stack([all_batches, last_of_all_batches], axis = 1)
	last_output_reshaped = tf.gather_nd(output_data, indices_list)
	last_output = tf.reshape(last_output_reshaped, [output_data_shape[0], 1, output_data_shape[2]])
	return last_output


soft_1_label = tf.random_uniform(shape = [BATCH_SIZE, 1], minval = 0.8, maxval = 1.2)
soft_0_label = tf.random_uniform(shape = [BATCH_SIZE, 1], minval = 0.0, maxval = 0.3)


# Define a noise generator
noise = tf.truncated_normal(dtype = tf.float32, mean = 0.0, stddev = 0.5, shape = [BATCH_SIZE, 1, CELL_SIZE], name = 'noise')
# Define the G_of_Z at the test time
GEN_pred_0 = tf.placeholder(tf.float32, [BATCH_SIZE, 1, CELL_SIZE], name = 'GEN_pred_0')
GEN_h_0 = tf.zeros([BATCH_SIZE, CELL_SIZE*STACK_SIZE])
[GEN_pred_1, GEN_h_1] = Generator(GEN_pred_0, GEN_h_0)
[GEN_pred_2, GEN_h_2] = Generator(GEN_pred_1, GEN_h_1)
[GEN_pred_3, GEN_h_3] = Generator(GEN_pred_2, GEN_h_2)
[GEN_pred_4, GEN_h_4] = Generator(GEN_pred_3, GEN_h_3)
[GEN_pred_5, GEN_h_5] = Generator(GEN_pred_4, GEN_h_4)
[GEN_pred_6, GEN_h_6] = Generator(GEN_pred_5, GEN_h_5)
GEN_pred_1_flat = tf.reshape(GEN_pred_1, [BATCH_SIZE*1, CELL_SIZE])
GEN_pred_2_flat = tf.reshape(GEN_pred_2, [BATCH_SIZE*1, CELL_SIZE])
GEN_pred_3_flat = tf.reshape(GEN_pred_3, [BATCH_SIZE*1, CELL_SIZE])
GEN_pred_4_flat = tf.reshape(GEN_pred_4, [BATCH_SIZE*1, CELL_SIZE])
GEN_pred_5_flat = tf.reshape(GEN_pred_5, [BATCH_SIZE*1, CELL_SIZE])
GEN_pred_6_flat = tf.reshape(GEN_pred_6, [BATCH_SIZE*1, CELL_SIZE])
recn_1 = DecoderPix2Pix(GEN_pred_1_flat, True)
recn_2 = DecoderPix2Pix(GEN_pred_2_flat, True)
recn_3 = DecoderPix2Pix(GEN_pred_3_flat, True)
recn_4 = DecoderPix2Pix(GEN_pred_4_flat, True)
recn_5 = DecoderPix2Pix(GEN_pred_5_flat, True)
recn_6 = DecoderPix2Pix(GEN_pred_6_flat, True)
GEN_pred_vid_1 = GEN_pred_1
GEN_pred_vid_2 = tf.concat([GEN_pred_vid_1, GEN_pred_2], axis = 1)
GEN_pred_vid_3 = tf.concat([GEN_pred_vid_2, GEN_pred_3], axis = 1)
GEN_pred_vid_4 = tf.concat([GEN_pred_vid_3, GEN_pred_4], axis = 1)
GEN_pred_vid_5 = tf.concat([GEN_pred_vid_4, GEN_pred_5], axis = 1)
DIS_GEN_vid_h_0 = tf.zeros([BATCH_SIZE, CELL_SIZE*STACK_SIZE])
[DIS_GEN_pred_vid_1, DIS_GEN_vid_h_1] = Discriminator(GEN_pred_vid_1, DIS_GEN_vid_h_0)
[DIS_GEN_pred_vid_2, DIS_GEN_vid_h_2] = Discriminator(GEN_pred_vid_2, DIS_GEN_vid_h_0)
[DIS_GEN_pred_vid_3, DIS_GEN_vid_h_3] = Discriminator(GEN_pred_vid_3, DIS_GEN_vid_h_0)
[DIS_GEN_pred_vid_4, DIS_GEN_vid_h_4] = Discriminator(GEN_pred_vid_4, DIS_GEN_vid_h_0)
[DIS_GEN_pred_vid_5, DIS_GEN_vid_h_5] = Discriminator(GEN_pred_vid_5, DIS_GEN_vid_h_0)
DIS_GEN_pred_vid_1_last = GetLastRNNOutput(DIS_GEN_pred_vid_1)
DIS_GEN_pred_vid_2_last = GetLastRNNOutput(DIS_GEN_pred_vid_2)
DIS_GEN_pred_vid_3_last = GetLastRNNOutput(DIS_GEN_pred_vid_3)
DIS_GEN_pred_vid_4_last = GetLastRNNOutput(DIS_GEN_pred_vid_4)
DIS_GEN_pred_vid_5_last = GetLastRNNOutput(DIS_GEN_pred_vid_5)
DIS_GEN_pred_vid_1_last_1 = tf.reshape(DIS_GEN_pred_vid_1_last, [BATCH_SIZE, CELL_SIZE])
DIS_GEN_pred_vid_2_last_1 = tf.reshape(DIS_GEN_pred_vid_2_last, [BATCH_SIZE, CELL_SIZE])
DIS_GEN_pred_vid_3_last_1 = tf.reshape(DIS_GEN_pred_vid_3_last, [BATCH_SIZE, CELL_SIZE])
DIS_GEN_pred_vid_4_last_1 = tf.reshape(DIS_GEN_pred_vid_4_last, [BATCH_SIZE, CELL_SIZE])
DIS_GEN_pred_vid_5_last_1 = tf.reshape(DIS_GEN_pred_vid_5_last, [BATCH_SIZE, CELL_SIZE])
DIS_fake_1 = tf.add(tf.matmul(DIS_GEN_pred_vid_1_last_1, DIS_W_dec), DIS_B_dec)
DIS_fake_2 = tf.add(tf.matmul(DIS_GEN_pred_vid_2_last_1, DIS_W_dec), DIS_B_dec)
DIS_fake_3 = tf.add(tf.matmul(DIS_GEN_pred_vid_3_last_1, DIS_W_dec), DIS_B_dec)
DIS_fake_4 = tf.add(tf.matmul(DIS_GEN_pred_vid_4_last_1, DIS_W_dec), DIS_B_dec)
DIS_fake_5 = tf.add(tf.matmul(DIS_GEN_pred_vid_5_last_1, DIS_W_dec), DIS_B_dec)
# D_of_fake loss
DIS_fake_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_fake_1, labels = soft_0_label))
DIS_fake_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_fake_2, labels = soft_0_label))
DIS_fake_loss_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_fake_3, labels = soft_0_label))
DIS_fake_loss_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_fake_4, labels = soft_0_label))
DIS_fake_loss_5 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_fake_5, labels = soft_0_label))
DIS_fake_loss = tf.add(DIS_fake_loss_1, tf.add(DIS_fake_loss_2, tf.add(DIS_fake_loss_3, tf.add(DIS_fake_loss_4, DIS_fake_loss_5))))


# G loss
GEN_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_fake_1, labels = soft_1_label))
GEN_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_fake_2, labels = soft_1_label))
GEN_loss_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_fake_3, labels = soft_1_label))
GEN_loss_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_fake_4, labels = soft_1_label))
GEN_loss_5 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_fake_5, labels = soft_1_label))
GEN_loss = tf.add(GEN_loss_1, tf.add(GEN_loss_2, tf.add(GEN_loss_3, tf.add(GEN_loss_4, GEN_loss_5))))


# Define D_of_X
X_real = tf.placeholder(tf.float32, [BATCH_SIZE, UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
X_real_reshaped = tf.reshape(X_im, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
X_real_encoded = EncoderPix2Pix(X_real_reshaped, True)
X_real_encoded_reshaped = tf.reshape(X_real_encoded, [BATCH_SIZE, UNFOLD_SIZE, CELL_SIZE])
GEN_true_vid_1 = tf.slice(X_real_encoded_reshaped, [0, 0, 0], [BATCH_SIZE, 1, CELL_SIZE])
GEN_true_vid_2 = tf.slice(X_real_encoded_reshaped, [0, 0, 0], [BATCH_SIZE, 2, CELL_SIZE])
GEN_true_vid_3 = tf.slice(X_real_encoded_reshaped, [0, 0, 0], [BATCH_SIZE, 3, CELL_SIZE])
GEN_true_vid_4 = tf.slice(X_real_encoded_reshaped, [0, 0, 0], [BATCH_SIZE, 4, CELL_SIZE])
GEN_true_vid_5 = tf.slice(X_real_encoded_reshaped, [0, 0, 0], [BATCH_SIZE, 5, CELL_SIZE])
DIS_GEN_h_0 = tf.zeros([BATCH_SIZE, CELL_SIZE*STACK_SIZE])
[DIS_GEN_true_vid_1, DIS_GEN_h_1] = Discriminator(GEN_true_vid_1, DIS_GEN_h_0)
[DIS_GEN_true_vid_2, DIS_GEN_h_2] = Discriminator(GEN_true_vid_2, DIS_GEN_h_0)
[DIS_GEN_true_vid_3, DIS_GEN_h_3] = Discriminator(GEN_true_vid_3, DIS_GEN_h_0)
[DIS_GEN_true_vid_4, DIS_GEN_h_4] = Discriminator(GEN_true_vid_4, DIS_GEN_h_0)
[DIS_GEN_true_vid_5, DIS_GEN_h_5] = Discriminator(GEN_true_vid_5, DIS_GEN_h_0)
DIS_GEN_true_vid_1_last = GetLastRNNOutput(DIS_GEN_true_vid_1)
DIS_GEN_true_vid_2_last = GetLastRNNOutput(DIS_GEN_true_vid_2)
DIS_GEN_true_vid_3_last = GetLastRNNOutput(DIS_GEN_true_vid_3)
DIS_GEN_true_vid_4_last = GetLastRNNOutput(DIS_GEN_true_vid_4)
DIS_GEN_true_vid_5_last = GetLastRNNOutput(DIS_GEN_true_vid_5)
DIS_GEN_true_vid_1_last_1 = tf.reshape(DIS_GEN_true_vid_1_last, [BATCH_SIZE, CELL_SIZE])
DIS_GEN_true_vid_2_last_1 = tf.reshape(DIS_GEN_true_vid_2_last, [BATCH_SIZE, CELL_SIZE])
DIS_GEN_true_vid_3_last_1 = tf.reshape(DIS_GEN_true_vid_3_last, [BATCH_SIZE, CELL_SIZE])
DIS_GEN_true_vid_4_last_1 = tf.reshape(DIS_GEN_true_vid_4_last, [BATCH_SIZE, CELL_SIZE])
DIS_GEN_true_vid_5_last_1 = tf.reshape(DIS_GEN_true_vid_5_last, [BATCH_SIZE, CELL_SIZE])
DIS_real_1 = tf.add(tf.matmul(DIS_GEN_true_vid_1_last_1, DIS_W_dec), DIS_B_dec)
DIS_real_2 = tf.add(tf.matmul(DIS_GEN_true_vid_2_last_1, DIS_W_dec), DIS_B_dec)
DIS_real_3 = tf.add(tf.matmul(DIS_GEN_true_vid_3_last_1, DIS_W_dec), DIS_B_dec)
DIS_real_4 = tf.add(tf.matmul(DIS_GEN_true_vid_4_last_1, DIS_W_dec), DIS_B_dec)
DIS_real_5 = tf.add(tf.matmul(DIS_GEN_true_vid_5_last_1, DIS_W_dec), DIS_B_dec)
# D_of_real loss
DIS_real_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_real_1, labels = soft_1_label))
DIS_real_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_real_2, labels = soft_1_label))
DIS_real_loss_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_real_3, labels = soft_1_label))
DIS_real_loss_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_real_4, labels = soft_1_label))
DIS_real_loss_5 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_real_5, labels = soft_1_label))
DIS_real_loss = tf.add(DIS_real_loss_1, tf.add(DIS_real_loss_2, tf.add(DIS_real_loss_3, tf.add(DIS_real_loss_4, DIS_real_loss_5))))

# Define for VAE
H_enc = EncoderPix2Pix(X_im_reshaped, True)
X_rec_reshaped = DecoderPix2Pix(H_enc, True)
X_rec = tf.reshape(X_rec_reshaped, [BATCH_SIZE, UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
X = tf.reshape(X_im_reshaped, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE*IMG_SIZE*ENC_INPUT_CHANNEL])
X_hat = tf.reshape(X_rec_reshaped, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE*IMG_SIZE*ENC_INPUT_CHANNEL])
# Linearize inputs and outputs
LLT = tf.reduce_sum(X*tf.log(X_hat + 1e-9) + (1 - X)*tf.log(1 - X_hat + 1e-9), axis = 1)
VLB = tf.reduce_mean(LLT)


####################################################################################################
##################################Saver and Initializer#############################################
####################################################################################################


# Define session and initializer
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# Define saver
saver = tf.train.Saver()


# Get all variables
trainable_variables_list = tf.trainable_variables()
print('[DEBUG] Trainable Variables List : ' + str(len(trainable_variables_list)))
for item in trainable_variables_list :
	print('[DEBUG] 		' + str(item.name))
# Get ENC variables
ENC_trainable_variables_list = [item for item in trainable_variables_list if 'ENC_' in item.name]
print('[DEBUG] ENC Trainable Variables List : ' + str(len(ENC_trainable_variables_list)))
for item in ENC_trainable_variables_list :
	print('[DEBUG] 		' + str(item.name))
# Get DEC variables
DEC_trainable_variables_list = [item for item in trainable_variables_list if 'DEC_' in item.name]
print('[DEBUG] DEC Trainable Variables List : ' + str(len(DEC_trainable_variables_list)))
for item in DEC_trainable_variables_list :
	print('[DEBUG] 		' + str(item.name))
# Get GEN variables
GEN_trainable_variables_list = [item for item in trainable_variables_list if 'GEN' in item.name]
print('[DEBUG] GEN Trainable Variables List : ' + str(len(GEN_trainable_variables_list)))
for item in GEN_trainable_variables_list :
	print('[DEBUG] 		' + str(item.name))
# Get DIS variables
DIS_trainable_variables_list = [item for item in trainable_variables_list if 'DIS' in item.name]
print('[DEBUG] DIS Trainable Variables List : ' + str(len(DIS_trainable_variables_list)))
for item in DIS_trainable_variables_list :
	print('[DEBUG] 		' + str(item.name))


print('[DEBUG] Exiting ... ')
sys.exit()


####################################################################################################
###########################DEFINE LOSS AND OPTIMIZERS FOR VAE#######################################
####################################################################################################


trainin_step = tf.train.AdamOptimizer().minimize(-VLB, var_list = ENC_trainable_variables_list + DEC_trainable_variables_list)
# L2_loss = tf.nn.l2_loss(X_hat - X)
# L2_training_step = tf.train.AdamOptimizer().minimize(L2_loss)
DIS_fake_trainin_step = tf.train.AdamOptimizer().minimize(DIS_fake_loss, var_list = GEN_trainable_variables_list + DIS_trainable_variables_list)
DIS_real_trainin_step = tf.train.AdamOptimizer().minimize(DIS_real_loss, var_list = GEN_trainable_variables_list + DIS_trainable_variables_list)
GEN_training_step = tf.train.AdamOptimizer().minimize(GEN_loss, var_list = GEN_trainable_variables_list + DIS_trainable_variables_list)


####################################################################################################
###############################VAE TRAINING ITERATIONS##############################################
####################################################################################################


for iteration in range(VAE_ITR) : 
	# print('[TRAINING] Iteration : ' + str(iteration))
	training_batch = data_loader.GetNextBatch()
	sess.run(trainin_step, feed_dict = { X_im : training_batch })
	if iteration%500 == 0 :
		print('[TRAINING] Iteration : ' + str(iteration))
		training_batch_display = data_loader.GetNextBatch()
		VLB_ = sess.run(VLB, feed_dict = { X_im : training_batch_display })
	if iteration%5000 == 0 :
		print('[TRAINING] Iteration : ' + str(iteration))
		print('[TRAINING] Saving Partial Model Reconstruction ... ')
		data_array = []
		training_batch_record = data_loader.GetNextBatch()
		[X_im_, X_rec_] = sess.run([X_im, X_rec], feed_dict = { X_im : training_batch_record })
		data_array.append(X_im_)
		data_array.append(X_rec_)
		data_array_np = np.array(data_array)
		np.save('VAE_Reconstructed_Dataset/CVRG_Reconstruction_Iteration_' + str(iteration), data_array_np)
		os.system('mkdir VAE_Models/Iteration_' + str(iteration))
		saver.save(sess, 'VAE_Models/Iteration_' + str(iteration) + '/Intermediate_Model_' + str(iteration))


for iteration1 in range(ITR) :
	# print('[TRAINING] Iteration : ' + str(iteration))
	training_batch = data_loader.GetNextBatch()
	sess.run(DIS_real_trainin_step, feed_dict = { X_real : training_batch })
	get_noise = sess.run(noise)
	sess.run(DIS_fake_trainin_step, feed_dict = { GEN_pred_0 : get_noise })
	get_noise = sess.run(noise)
	sess.run(GEN_trainin_step, feed_dict = { GEN_pred_0 : get_noise })
	if iteration1%100 == 0 :
		print('[TRAINING] Iteration : ' + str(iteration))
	if iteration1%500 == 0 :
		get_noise = sess.run(noise)
		training_batch = data_loader.GetNextBatch()
		DIS_real_loss_ = sess.run(DIS_real_loss, feed_dict = { X_real : training_batch })
		print('[TRAINING] DIS_real_loss : ' + str(DIS_real_loss_))
		DIS_fake_loss_ = sess.run(DIS_fake_loss, feed_dict = { GEN_pred_0 : get_noise })
		print('[TRAINING] DIS_fake_loss : ' + str(DIS_fake_loss_))
		GEN_loss_ = sess.run(GEN_loss, feed_dict = { GEN_pred_0 : get_noise })
		print('[TRAINING] GEN_loss : ' + str(GEN_loss_))
	if iteration1%2500 == 0 :
		data = []
		print('[TRAINING] Generating Projectiles ... ')
		for j in range(32) :
			data.append([])
			get_noise = sess.run(noise)
			[recn_1_, recn_2_, recn_3_, recn_4_, recn_5_, recn_6_] = sess.run([recn_1, recn_2, recn_3, recn_4, recn_5, recn_6], feed_dict = { GEN_pred_0 : get_noise })
			data[-1].append(recn_1_)
			data[-1].append(recn_2_)
			data[-1].append(recn_3_)
			data[-1].append(recn_4_)
			data[-1].append(recn_5_)
			data[-1].append(recn_6_)
		data_np = np.array(data)
		np.save('C_RNN_GAN_Generated_Dataset/Iteration_' + str(iteration1), data_np)
		os.system('mkdir C_RNN_GAN_Models/Iteration_' + str(iteration))
		saver.save(sess, 'C_RNN_GAN_Models/Iteration_' + str(iteration) + '/Intermediate_Model_' + str(iteration))

			


