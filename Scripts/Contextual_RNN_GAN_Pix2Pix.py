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
BATCH_SIZE = 64
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
UNFOLD_SIZE = 5
# Auxiliary Parameters
LABEL_SIZE = 2 # x, y of patch for now.. Can vary with the question
# Iterations
ITR = 100000
# Training Parameters
lr_GradientDescentOptimizer = 0.00001
lr_AdamOptimizer = 0.001


# We must implement GAN-training hacks for this work
# https://github.com/soumith/ganhacks


####################################################################################################
########################################DATA LOADER#################################################
####################################################################################################


class dataLoader :
	"""
	The data loader class
	"""
	# Define the constructor
	def __init__(self, tr_prop = 0.60, val_prop = 0.20, te_prop = 0.20):
		# Information message
		print('[INFO] Creating Data Loader Instance ...')
		self.data_dir = 'Dataset/'
		self.dir_list = os.listdir(self.data_dir)
		#print '[DEBUG] dir_list : ', self.dir_list
		self.train_dir_list = []
		self.validation_dir_list = []
		self.test_dir_list = []
		# Randomly distribute the datasets as train, validation and test
		for a_dir in self.dir_list :
			r = np.random.random()
			if r <= tr_prop :
				self.train_dir_list.append(a_dir)
			elif r <= tr_prop + val_prop:
				self.validation_dir_list.append(a_dir)
			else :
				self.test_dir_list.append(a_dir)
		# Very rare error!!
		if len(self.train_dir_list) == 0 or len(self.validation_dir_list) == 0 or len(self.test_dir_list) == 0:
			print('[ERROR] One of the data splits is empty!!')
			sys.exit()
		#print '[DEBUG] Training data split : ', self.train_dir_list # DO NOT FORGET TO REMOVE .m FILES!!
		print('[INFO] Data Loader Initiated ...')


	# Define a function to get the next batch
	def GetNextBatch(self, batch_size = BATCH_SIZE): 
		# Define batches
		batch_im = []
		# Select the directories from which to get sequences
		for a_batch in range(batch_size):
			batch_im.append([])
			a_dir = random.choice(self.train_dir_list)
		    print(str(a_dir))
			# open the dir
			this_dir = self.data_dir + a_dir
			im_list = os.listdir(this_dir)
			# length from first 50 or so
			first = np.random.randint(1, 10)
			diff = np.random.randint(1, 10)
			# Add videos
			for _ in range(5):
				next_im = cv2.imread(this_dir + '/' + im_list[first], cv2.IMREAD_GRAYSCALE)
				next_im = np.reshape(next_im, [IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
				batch_im[-1].append(next_im)
				first = first + diff
			# Print size of the batch
		batch_im_np = np.array(batch_im)
		#print '[DEBUG] batch_im_np shape : ', batch_im_np.shape
		return batch_im_np


# Generate a data loader instance
data_loader = dataLoader()
next_batch = data_loader.GetNextBatch()
sys.exit()


####################################################################################################
#################################PLACEHOLDERS AND VARIABLES#########################################
####################################################################################################


# Define placeholders and variables
# Input image placeholders
X_im = tf.placeholder(tf.float32, [BATCH_SIZE, UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL], name = 'X_im') # [BATCH_SIZE, UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL]
X_label = tf.placeholder(tf.float32, [BATCH_SIZE, UNFOLD_SIZE, LABEL_SIZE], name = 'X_label') # [BATCH_SIZE, UNFOLD_SIZE, LABEL_SIZE]
X_im_reshaped = tf.reshape(X_im, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL]) # [BATCH_SIZE, UNFOLD_SIZE, LABEL_SIZE]
X_label_reshaped = tf.reshape(X_label, [BATCH_SIZE*UNFOLD_SIZE, LABEL_SIZE]) # [BATCH_SIZE, UNFOLD_SIZE, LABEL_SIZE]
# Encoder variables
ENC_W1 = tf.Variable(tf.truncated_normal([6, 6, ENC_INPUT_CHANNEL, ENC_K1], stddev = 0.1), name = 'ENC_W1')
ENC_B1 = tf.Variable(tf.zeros([ENC_K1]), name = 'ENC_B1')
ENC_W2 = tf.Variable(tf.truncated_normal([6, 6, ENC_K1, ENC_K2], stddev = 0.1), name = 'ENC_W2')
ENC_B2 = tf.Variable(tf.zeros([ENC_K2]), name = 'ENC_B2')
ENC_W3 = tf.Variable(tf.truncated_normal([6, 6, ENC_K2, ENC_K3], stddev = 0.1), name = 'ENC_W3')
ENC_B3 = tf.Variable(tf.zeros([ENC_K3]), name = 'ENC_B3')
ENC_W4 = tf.Variable(tf.truncated_normal([HIDDEN_DIM*HIDDEN_DIM*ENC_K3, LATENT_DIM], stddev = 0.1), name = 'ENC_W4')
ENC_B4 = tf.Variable(tf.zeros([LATENT_DIM]), name = 'ENC_B4')
# Decoder variables
DEC_W1 = tf.Variable(tf.truncated_normal([LATENT_DIM, HIDDEN_DIM*HIDDEN_DIM*ENC_K3], stddev = 0.1), name = 'DEC_W1')
DEC_B1 = tf.Variable(tf.zeros([HIDDEN_DIM*HIDDEN_DIM*ENC_K3]), name = 'DEC_B1')
DEC_W2 = tf.Variable(tf.truncated_normal([6, 6, DEC_K1, DEC_K2], stddev = 0.1), name = 'DEC_W2')
DEC_B2 = tf.Variable(tf.zeros([DEC_K2]), name = 'DEC_B2')
DEC_W3 = tf.Variable(tf.truncated_normal([6, 6, DEC_K2, DEC_K3], stddev = 0.1), name = 'DEC_W3')
DEC_B3 = tf.Variable(tf.zeros([DEC_K3]), name = 'DEC_B3')
DEC_W4 = tf.Variable(tf.truncated_normal([6, 6, DEC_K3, DEC_OUTPUT_CHANNEL], stddev = 0.1), name = 'DEC_W4')
DEC_B4 = tf.Variable(tf.zeros([DEC_OUTPUT_CHANNEL]), name = 'DEC_B4')
# Sampler variables
SAM_W_mu = tf.Variable(tf.truncated_normal([LATENT_DIM, LATENT_DIM], stddev = 0.1), name = 'SAM_W_mu')
SAM_B_mu = tf.Variable(tf.zeros([LATENT_DIM]), name = 'SAM_B_mu')
SAM_W_logstd = tf.Variable(tf.truncated_normal([LATENT_DIM, LATENT_DIM], stddev = 0.1), name = 'SAM_W_logstd')
SAM_B_logstd = tf.Variable(tf.zeros([LATENT_DIM]), name = 'SAM_B_logstd')
# Auxiliary network variables
AUX_W_aux = tf.Variable(tf.truncated_normal([LATENT_DIM, LABEL_SIZE], stddev = 0.1), name = 'AUX_W_aux')
AUX_B_aux = tf.Variable(tf.zeros([LABEL_SIZE]), name = 'AUX_B_aux')
# Decoder single layer neuron
DIS_W_dec = tf.Variable(tf.truncated_normal([LATENT_DIM, 1], stddev = 0.1), name = 'DIS_W_dec')
DIS_B_dec = tf.Variable(tf.zeros([1]), name = 'DIS_B_dec')
# RNN definitions: GEN for generator and DIS for discriminator
GEN_H_in = tf.placeholder(tf.float32, [BATCH_SIZE, CELL_SIZE*STACK_SIZE], name = 'GEN_H_in')
GEN_gru_cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
GEN_multi_gru_cell = tf.contrib.rnn.MultiRNNCell([GEN_gru_cell]*STACK_SIZE, state_is_tuple = False)
DIS_H_in = tf.placeholder(tf.float32, [BATCH_SIZE, CELL_SIZE*STACK_SIZE], name = 'DIS_H_in')
DIS_gru_cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
DIS_multi_gru_cell = tf.contrib.rnn.MultiRNNCell([DIS_gru_cell]*STACK_SIZE, state_is_tuple = False)


####################################################################################################
##########################################ENCODER###################################################
####################################################################################################


# Define the encoder's forward pass
def Encoder(x, reuse = None):
	#x_ = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, ENC_INPUT_CHANNEL])
	ENC_Y1_2 = tf.add(tf.nn.conv2d(x, ENC_W1, strides = [1, 2, 2, 1], padding = 'SAME'), ENC_B1)
	#ENC_Y2 = tf.layers.batch_normalization(ENC_Y1, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'ENC_BN1')
	ENC_Y3 = tf.nn.relu(ENC_Y1_2)
	ENC_Y3_1 = tf.nn.avg_pool(ENC_Y3, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	ENC_Y4_5 = tf.add(tf.nn.conv2d(ENC_Y3_1, ENC_W2, strides = [1, 2, 2, 1], padding = 'SAME'), ENC_B2)
	#ENC_Y5 = tf.layers.batch_normalization(ENC_Y4, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'ENC_BN2')
	ENC_Y6 = tf.nn.relu(ENC_Y4_5)
	ENC_Y6_1 = tf.nn.avg_pool(ENC_Y6, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
	ENC_Y7_8 = tf.add(tf.nn.conv2d(ENC_Y6_1, ENC_W3, strides = [1, 2, 2, 1], padding = 'SAME'), ENC_B3)
	#ENC_Y8 = tf.layers.batch_normalization(ENC_Y7, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'ENC_BN3')
	ENC_Y9 = tf.nn.relu(ENC_Y7_8)
	ENC_Y10 =  tf.reshape(ENC_Y9, [-1, HIDDEN_DIM*HIDDEN_DIM*ENC_K3]) # One 2 per each dimension reduction of 2, flatten out and return
	ENC_Y11 = tf.tanh(tf.add(tf.matmul(ENC_Y10, ENC_W4), ENC_B4)) # This can be kept without tanh, we are adding as a mimic step for VAE training.
	# print '[DEBUG] ENC Y9 shape : ', ENC_Y9.shape
	# print '[DEBUG] ENC Y10 shape : ', ENC_Y10.shape
	# print '[DEBUG] ENC Y11 shape : ', ENC_Y11.shape
	return ENC_Y11


## To call basically do this: 
# m_opts = argparser()
# m_vars = {}
# if m_opts['mode'] == 'test':
#	m_vars['images'] = tf.placeholder('float32', shape=[m_opts['img_height'],m_opts['img_width'],3], name='input_images')
#	m_vars['images'] = tf.expand_dims(m_vars['images'],0)
# if m_opts['mode'] == 'train':
#	m_vars = {}
#	m_vars['images'] = tf.placeholder('float32',
#	                                  shape=[None,None,6],
#	                                  name='input_images')
#
#	m_vars['queue'] = tf.FIFOQueue(10000,[tf.float32],
#	                               shapes=[[m_opts['img_height'],m_opts['img_width'],6]])
#
#	m_vars['enqueue_op'] = m_vars['queue'].enqueue([m_vars['images']])
#	m_vars['batch_images'] = m_vars['queue'].dequeue_many(m_opts['batch_size'])
# 
# m_vars = Encoder1(m_opts, m_vars)
# m_vars = Decoder1(m_opts, m_vars)
# 

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def lrelu(x,leak=0.2):
    return tf.maximum(x,leak*x)

def get_conv_filter(name,shape):
    return tf.get_variable(
           name=name+'_filter',
           shape=shape,
           initializer=tf.contrib.layers.xavier_initializer(uniform=True,
                                                            seed=None,
                                                            dtype=tf.float32))


def get_conv_bias(name,shape):
    return tf.get_variable(
           name=name+'_bias',
           shape=shape,
           initializer=tf.constant_initializer(0.0))


def conv_layer(bottom,name,shape,strides=[1,2,2,1],padding='SAME',
               bn=True,nonlin=True,is_train=True):
    '''
    bn_name will be used to use different bn params but shared weight params
    bn to regulate application of batch_norm
    nonlin to regulate application of non linearity
    '''
    # with tf.device('/cpu:0'):
    filt = get_conv_filter(name,shape)

    conv = tf.nn.conv2d(bottom, filt, strides, padding=padding)

    if bn:
        bn = batch_norm(name='bn_'+name)
        bias = bn(conv,train=is_train)

    else:
        # with tf.device('/cpu:0'):
        conv_bias = get_conv_bias(name,[shape[-1]])
        bias = tf.nn.bias_add(conv,conv_bias)

    if nonlin:
        relu = lrelu(bias)

    else:
        relu = bias

    return relu

def uconv_layer(bottom,name,shape,output_shape,strides=[1,2,2,1],
                padding='SAME',bn=True,nonlin=True,is_train=True):
    '''
    bn to regulate application of batch_norm
    nonlin to regulate application of non linearity
    '''
    with tf.device('/cpu:0'):
        filt = get_conv_filter(name,shape)

    conv = tf.nn.conv2d_transpose(bottom,filt,output_shape,[1,2,2,1], padding=padding)

    if bn:
        bn = batch_norm(name='bn_'+name)
        bias = bn(conv,train=is_train)

    else:
        with tf.device('/cpu:0'):
            conv_bias = get_conv_bias(name,[output_shape[-1]])
        bias = tf.nn.bias_add(conv,conv_bias)

    if nonlin:
        relu = lrelu(bias)

    else:
        relu = bias

    return relu


def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-name', type=str, default='Physics-GAN')
    parser.add_argument('-mode', type=str, default='train')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-print_freq', type=int, default=50)
    parser.add_argument('-save_freq', type=int, default=5000)
    parser.add_argument('-bottleneck', type=int, default=256)
    parser.add_argument('-data_dir', type=str, default='/root/workspace/dataset/')
    parser.add_argument('-save', type=str, default='true')
    parser.add_argument('-step', type=int, default=15000)
    parser.add_argument('-save_dir', type=str, default='../../datasets/')
    parser.add_argument('-img_height', type=int, default=224)
    parser.add_argument('-img_width', type=int, default=224)

    m_opts = vars(parser.parse_args())
    return m_opts

def Encoder1(m_opts, m_vars):
    with tf.variable_scope('encoder') as scope:
        is_train = m_opts['mode']=='train'

        if m_opts['multi_scale']=='false':
            if is_train:
            	# Batch, y, x, dimension
            	# in_images is a single image of 3 dimensions
                m_vars['in_images'] = transform(m_vars['batch_images'][:,:,:,:3])
                m_vars['out_image'] = transform(m_vars['batch_images'][:,:,:,3:])
    
            else:
                m_vars['in_images'] = transform(m_vars['images'])
                # Dummy to make the output_shape for the last decoder layer work
                m_vars['out_image'] = transform(m_vars['images'][:,:,:3])
    
            m_vars['conv_in'] = conv_layer(m_vars['in_images'], 'conv_in',
                                           [4,4,3,64], strides=[1,1,1,1],
                                           is_train=is_train)
    
            m_vars['conv_1'] = conv_layer(m_vars['conv_in'], 'conv_1',
                                          [4,4,64,128], is_train=is_train)
            if m_opts['bottleneck'] == 128:
                m_vars['conv_2'] = conv_layer(m_vars['conv_1'], 'conv_2',
                                              [4,4,128,128], is_train=is_train)
                # conv_3 is bottle neck layer
                m_vars['conv_3'] = conv_layer(m_vars['conv_2'], 'conv_3',
                                              [4,4,128,128], is_train=is_train)
    
            if m_opts['bottleneck'] == 256:
                m_vars['conv_2'] = conv_layer(m_vars['conv_1'], 'conv_2',
                                              [4,4,128,256], is_train=is_train)
                m_vars['conv_3'] = conv_layer(m_vars['conv_2'], 'conv_3',
                                              [4,4,256,256], is_train=is_train)
           

    return m_vars

def Decoder1(m_opts, m_vars):
    # In uconv filter last two indices have swapped meaning in comparison to conv
    # conv2d -> [filter_height, filter_width, in_channels, out_channels]
    # conv2d_transpose (used in uconv) -> [filter_height, filter_width, output_channels, input_channels]
    with tf.variable_scope('decoder'):
        is_train = m_opts['mode']=='train'

        if m_opts['bottleneck'] == 128:
            m_vars['uconv_1'] = uconv_layer(m_vars['conv_3'], 'uconv_1',
                                            [4,4,128,128],
                                            tf.shape(m_vars['conv_2']),
                                            is_train=is_train)
            m_vars['uconv_1'] = tf.concat([m_vars['uconv_1'],m_vars['conv_2']],
                                          axis=3)
            m_vars['uconv_2'] = uconv_layer(m_vars['uconv_1'], 'uconv_2',
                                            [4,4,128,256],
                                            tf.shape(m_vars['conv_1']),
                                            is_train=is_train)
            m_vars['uconv_2'] = tf.concat([m_vars['uconv_2'],m_vars['conv_1']],
                                          axis=3)
            m_vars['uconv_3'] = uconv_layer(m_vars['uconv_2'], 'uconv_3',
                                            [4,4,64,256],
                                            tf.shape(m_vars['conv_in']),
                                            is_train=is_train)
            m_vars['uconv_3'] = tf.concat([m_vars['uconv_3'],m_vars['conv_in']],
                                           axis=3)
            
            m_vars['conv_out'] = conv_layer(m_vars['uconv_3'], 'conv_out',
                                            [4,4,128,3], strides=[1,1,1,1],
                                            nonlin=False, bn=False,
                                            is_train=is_train)     

        if m_opts['bottleneck'] == 256:
            m_vars['uconv_1'] = uconv_layer(m_vars['conv_3'], 'uconv_1',
                                            [4,4,256,256],
                                            tf.shape(m_vars['conv_2']),
                                            is_train=is_train)
            m_vars['uconv_1'] = tf.concat([m_vars['uconv_1'],m_vars['conv_2']],
                                          axis=3)
            m_vars['uconv_2'] = uconv_layer(m_vars['uconv_1'], 'uconv_2',
                                            [4,4,128,512],
                                            tf.shape(m_vars['conv_1']),
                                            is_train=is_train)
            m_vars['uconv_2'] = tf.concat([m_vars['uconv_2'],m_vars['conv_1']],
                                          axis=3)
            m_vars['uconv_3'] = uconv_layer(m_vars['uconv_2'], 'uconv_3',
                                            [4,4,64,256],
                                            tf.shape(m_vars['conv_in']),
                                            is_train=is_train)
            m_vars['uconv_3'] = tf.concat([m_vars['uconv_3'],m_vars['conv_in']],
                                           axis=3)
            
            m_vars['conv_out'] = conv_layer(m_vars['uconv_3'], 'conv_out',
                                            [4,4,128,3], strides=[1,1,1,1],
                                            nonlin=False, bn=False,
                                            is_train=is_train)

            # m_vars['dec_image'] = tf.tanh(m_vars['conv_out'])
            m_vars['output_image'] = tf.tanh(m_vars['conv_out'])

    return m_vars
####################################################################################################
##########################################SAMPLER###################################################
####################################################################################################


# Define the sampler's forward pass # OMIT FOR NOW!! It is not fixed if we wish to make the network variational
def Sampler(x, reuse = None):	
	SAM_mu = tf.add(tf.matmul(x, SAM_W_mu), SAM_B_mu)
	SAM_logstd = tf.add(tf.matmul(x, SAM_W_logstd), SAM_B_logstd)
	SAM_noise = tf.random_normal([1, LATENT_DIM])
	SAM_Z = SAM_mu + tf.multiply(SAM_noise, tf.exp(0.5*SAM_logstd))
	#print '[DEBUG] Sampled vector shape : ', SAM_Z.shape
	return SAM_mu, SAM_logstd, SAM_Z


####################################################################################################
##########################################DECODER###################################################
####################################################################################################


# Define the decoder's forward pass
def Decoder(x, reuse = None):
	DEC_Y1 = tf.add(tf.matmul(x, DEC_W1), DEC_B1)
	DEC_Y2 = tf.reshape(DEC_Y1, [-1, HIDDEN_DIM, HIDDEN_DIM, DEC_K1])
	DEC_Y2_1 = tf.image.resize_images(DEC_Y2, [int(IMG_SIZE/2), int(IMG_SIZE/2)])
	DEC_Y3 = tf.layers.batch_normalization(DEC_Y2_1, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'DEC_BN1')
	DEC_Y4 = tf.nn.relu(DEC_Y3)
	DEC_Y5 = tf.add(tf.nn.conv2d(DEC_Y4, DEC_W2, strides = [1, 2, 2, 1], padding = 'SAME'), DEC_B2)
	DEC_Y6 = tf.layers.batch_normalization(DEC_Y5, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'DEC_BN2')
	DEC_Y7 = tf.nn.relu(DEC_Y6)
	DEC_Y8 = tf.image.resize_images(DEC_Y7, [IMG_SIZE, IMG_SIZE])
	DEC_Y9 = tf.add(tf.nn.conv2d(DEC_Y8, DEC_W3, strides = [1, 2, 2, 1], padding = 'SAME'), DEC_B3)
	DEC_Y10 = tf.layers.batch_normalization(DEC_Y9, epsilon = 1e-5, reuse = reuse, trainable = True, name = 'DEC_BN3')
	DEC_Y11 = tf.nn.relu(DEC_Y10)
	DEC_Y12 = tf.image.resize_images(DEC_Y11, [IMG_SIZE*2, IMG_SIZE*2])
	DEC_Y13 = tf.add(tf.nn.conv2d(DEC_Y12, DEC_W4, strides = [1, 2, 2, 1], padding = 'SAME'), DEC_B4)
	DEC_Y14 = tf.nn.sigmoid(DEC_Y13)
	return DEC_Y14


####################################################################################################
#########################################GENERATOR##################################################
####################################################################################################


# Define the dynamic recurrent forward pass of generator : We will use GRU/LSTM
def Generator(x, GEN_h_in):
	#GEN_X_ = tf.reshape(x, [BATCH_SIZE, UNFOLD_SIZE, CELL_SIZE])
	#GEN_H_r, GEN_H = tf.nn.dynamic_rnn(GEN_multi_gru_cell, GEN_X_, initial_state = GEN_h_in)
	GEN_H_r, GEN_H = tf.nn.dynamic_rnn(GEN_multi_gru_cell, x, initial_state = GEN_h_in, scope = 'GEN')
	GEN_H_r_norm = tf.tanh(GEN_H_r)
	return GEN_H_r_norm, GEN_H


####################################################################################################
#######################################DISCRIMINATOR################################################
####################################################################################################


# Define the dynamic recurrent forward pass of discriminator: We will use GRU/LSTM
def Discriminator(x, DIS_h_in):
	#DIS_X_ = tf.reshape(x, [BATCH_SIZE, UNFOLD_SIZE, CELL_SIZE])
	#DIS_H_r, DIS_H = tf.nn.dynamic_rnn(DIS_multi_gru_cell, DIS_X_, initial_state = DIS_h_in)
	DIS_H_r, DIS_H = tf.nn.dynamic_rnn(DIS_multi_gru_cell, x, initial_state = DIS_h_in, scope = 'DIS')
	return DIS_H_r, DIS_H


####################################################################################################
#########################################AUXILIARY##################################################
####################################################################################################


# Define the auxiliary network # Define custom net here
def Auxiliary(x):
	AUX_Y1 = tf.add(tf.matmul(x, AUX_W_aux), AUX_B_aux)
	return AUX_Y1


####################################################################################################
######################################TRIALS : IGNORE###############################################
####################################################################################################


# Define trial functions so that reuse = True works well later!!
encoded_vector_trial = Encoder(X_im_reshaped, None) # Has size of [BATCH_SIZE, LATENT_DIM]
#print '[INFO] Encoded vector trial size : ', encoded_vector_trial.shape
[sampled_mean_trial, sampled_logstd_trial, sampled_vector_trial] = Sampler(encoded_vector_trial)
decoded_vector_trial = Decoder(sampled_vector_trial)


####################################################################################################
############################ENCODE-SAMPLE-DECODE FORWARD PASS#######################################
####################################################################################################


# Define the forward pass through variational autoencoder for the input images
encoded_vector_reshaped = Encoder(X_im_reshaped, reuse = True)
encoded_vector = tf.reshape(encoded_vector_reshaped, [BATCH_SIZE, UNFOLD_SIZE, LATENT_DIM])
[sampled_mean_reshaped, sampled_logstd_reshaped, sampled_vector_reshaped] = Sampler(encoded_vector_reshaped, reuse = True)
# This may be required to pass to RNNs instead of passing the encoded vector reshaped
sampled_vector = tf.reshape(sampled_vector_reshaped, [BATCH_SIZE, UNFOLD_SIZE, LATENT_DIM])
sampled_mean = tf.reshape(sampled_mean_reshaped, [BATCH_SIZE, UNFOLD_SIZE, LATENT_DIM])
reconstructed_im_reshaped = Decoder(sampled_vector_reshaped, reuse = True)
reconstructed_im = tf.reshape(reconstructed_im_reshaped, [BATCH_SIZE, UNFOLD_SIZE, IMG_SIZE, IMG_SIZE, DEC_OUTPUT_CHANNEL])
# Predict the auxiliary labels
predicted_label_reshaped = Auxiliary(sampled_mean_reshaped)


####################################################################################################
###################################VAE LOSS DEFINITIONS#############################################
####################################################################################################


# Define VAE loss over the process of auto-encoding
X_im_lin = tf.reshape(X_im_reshaped, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE*IMG_SIZE])
reconstructed_im_lin = tf.reshape(reconstructed_im_reshaped, [BATCH_SIZE*UNFOLD_SIZE, IMG_SIZE*IMG_SIZE])
LL_term = tf.reduce_sum(X_im_lin*tf.log(reconstructed_im_lin + 1e-9) + (1 - X_im_lin)*tf.log(1 - reconstructed_im_lin + 1e-9), axis = 1)
KL_term = -0.5*tf.reduce_sum(1 + 2*sampled_logstd_reshaped - tf.pow(sampled_mean_reshaped, 2) - tf.exp(2*sampled_logstd_reshaped), axis = 1)
variational_lower_bound_loss = tf.reduce_mean(LL_term - KL_term)
aux_pred_loss = tf.nn.l2_loss(predicted_label_reshaped - X_label_reshaped, name = 'aux_pred_term')


# Optimizers
optimize_VAE_Loss = tf.train.AdamOptimizer(lr_AdamOptimizer).minimize(-variational_lower_bound_loss)
# We avoid the L_p loss for now


####################################################################################################
##############################ONE-BY-ONE ENC-DEC FORWARD PASS#######################################
####################################################################################################


# This is the first type of RNN # Define the forward path through generator as one-by-one image prediction
GEN_pred_0 = tf.truncated_normal(dtype = tf.float32, mean = 0.0, stddev = 0.5, shape = [BATCH_SIZE, 1, LATENT_DIM], name = 'GEN_pred_0') # Pass noise here!! stddev = 0.5 because -1 to 1 entries are desired for the noise
GEN_h_0 = tf.zeros([BATCH_SIZE, CELL_SIZE*STACK_SIZE])
#print '[DEBUG] ################################################################################'
#print '[DEBUG] ################################################################################'
# This is mainly for generating images one-by-one
[GEN_pred_1, GEN_h_1] = Generator(GEN_pred_0, GEN_h_0)
GEN_pred_1_reshaped = tf.reshape(GEN_pred_1, [-1, CELL_SIZE])
GEN_mean_1, _, _ = Sampler(GEN_pred_1_reshaped)
reconstr_1 = Decoder(GEN_mean_1, reuse = True)
#print '[DEBUG] GEN pred 1 shape : ', GEN_pred_1.shape, ' GEN h 1 shape : ', GEN_h_1.shape
[GEN_pred_2, GEN_h_2] = Generator(GEN_pred_1, GEN_h_1)
GEN_pred_2_reshaped = tf.reshape(GEN_pred_2, [-1, CELL_SIZE])
GEN_mean_2, _, _ = Sampler(GEN_pred_2_reshaped)
reconstr_2 = Decoder(GEN_mean_2, reuse = True)
#print '[DEBUG] GEN pred 2 shape : ', GEN_pred_2.shape, ' GEN h 2 shape : ', GEN_h_2.shape
[GEN_pred_3, GEN_h_3] = Generator(GEN_pred_2, GEN_h_2)
GEN_pred_3_reshaped = tf.reshape(GEN_pred_3, [-1, CELL_SIZE])
GEN_mean_3, _, _ = Sampler(GEN_pred_3_reshaped)
reconstr_3 = Decoder(GEN_mean_3, reuse = True)
#print '[DEBUG] GEN pred 3 shape : ', GEN_pred_3.shape, ' GEN h 3 shape : ', GEN_h_3.shape
[GEN_pred_4, GEN_h_4] = Generator(GEN_pred_3, GEN_h_3)
GEN_pred_4_reshaped = tf.reshape(GEN_pred_4, [-1, CELL_SIZE])
GEN_mean_4, _, _ = Sampler(GEN_pred_4_reshaped)
reconstr_4 = Decoder(GEN_mean_4, reuse = True)
#print '[DEBUG] GEN pred 4 shape : ', GEN_pred_4.shape, ' GEN h 4 shape : ', GEN_h_4.shape
[GEN_pred_5, GEN_h_5] = Generator(GEN_pred_4, GEN_h_4)
GEN_pred_5_reshaped = tf.reshape(GEN_pred_5, [-1, CELL_SIZE])
GEN_mean_5, _, _ = Sampler(GEN_pred_5_reshaped)
reconstr_5 = Decoder(GEN_mean_5, reuse = True)
#print '[DEBUG] GEN pred 5 shape : ', GEN_pred_5.shape, ' GEN h 5 shape : ', GEN_h_5.shape
# Define predicted image as well!
[GEN_pred_6, GEN_h_6] = Generator(GEN_pred_5, GEN_h_5)
GEN_pred_6_reshaped = tf.reshape(GEN_pred_6, [-1, CELL_SIZE])
GEN_mean_6, _, _ = Sampler(GEN_pred_6_reshaped)
reconstr_6 = Decoder(GEN_mean_6, reuse = True)
#print '[DEBUG] GEN pred 6 shape : ', GEN_pred_6.shape, ' GEN h 6 shape : ', GEN_h_6.shape
# Define discriminator
[DIS_GEN_pred_1, DIS_GEN_h_1] = Discriminator(GEN_pred_1, GEN_h_0)
DIS_GEN_pred_1_1 = tf.reshape(DIS_GEN_pred_1, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_pred_1_1 shape : ', DIS_GEN_pred_1_1.shape, ' DIS_GEN_h_1 shape : ', DIS_GEN_h_1.shape
[DIS_GEN_pred_2, DIS_GEN_h_2] = Discriminator(GEN_pred_2, GEN_h_0)
DIS_GEN_pred_2_1 = tf.reshape(DIS_GEN_pred_2, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_pred_2_1 shape : ', DIS_GEN_pred_2_1.shape, ' DIS_GEN_h_2 shape : ', DIS_GEN_h_2.shape
[DIS_GEN_pred_3, DIS_GEN_h_3] = Discriminator(GEN_pred_3, GEN_h_0)
DIS_GEN_pred_3_1 = tf.reshape(DIS_GEN_pred_3, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_pred_3_1 shape : ', DIS_GEN_pred_3_1.shape, ' DIS_GEN_h_3 shape : ', DIS_GEN_h_3.shape
[DIS_GEN_pred_4, DIS_GEN_h_4] = Discriminator(GEN_pred_4, GEN_h_0)
DIS_GEN_pred_4_1 = tf.reshape(DIS_GEN_pred_4, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_pred_4_1 shape : ', DIS_GEN_pred_4_1.shape, ' DIS_GEN_h_4 shape : ', DIS_GEN_h_4.shape
[DIS_GEN_pred_5, DIS_GEN_h_5] = Discriminator(GEN_pred_5, GEN_h_0)
DIS_GEN_pred_5_1 = tf.reshape(DIS_GEN_pred_5, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_pred_5_1 shape : ', DIS_GEN_pred_5_1.shape, ' DIS_GEN_h_5 shape : ', DIS_GEN_h_5.shape


####################################################################################################
#############################SELF CONTEXT ENC-DEC FORWARD PASS######################################
####################################################################################################


# This is the secod type of RNN # Define the forward path through generator as one-by-one image prediction WITH CONTEXT appended
GEN_video_0 = tf.truncated_normal(dtype = tf.float32, mean = 0.0, stddev = 0.5, shape = [BATCH_SIZE, 1, LATENT_DIM], name = 'GEN_video_0') # Pass the initial noise here
GEN_h_video_const = tf.zeros([BATCH_SIZE, CELL_SIZE*STACK_SIZE]) # This will be constant
# Define the function that extracts the last output
def GetLastRNNOutput(output_data): # [BATCH_SIZE, None, CELL_SIZE] shaped with dynamic None
	output_data_shape = tf.shape(output_data)
	#print '[DEBUG] output_data_shape : ', output_data_shape[0], output_data_shape[1], output_data_shape[2]
	all_batches = tf.range(output_data_shape[0])
	last_of_all_batches = tf.fill([output_data_shape[0]], output_data_shape[1]-1)
	indices_list = tf.stack([all_batches, last_of_all_batches], axis = 1)
	last_output_reshaped = tf.gather_nd(output_data, indices_list)
	#print '[DEBUG] last_output_reshaped shape : ', last_output_reshaped.shape
	last_output = tf.reshape(last_output_reshaped, [output_data_shape[0], 1, output_data_shape[2]])
	#print '[DEBUG] last_output shape : ', last_output.shape
	return last_output
# Start the forward pass
#print '[DEBUG] ################################################################################'
#print '[DEBUG] ################################################################################'
[GEN_video_pred_1, GEN_video_h_1] = Generator(GEN_video_0, GEN_h_video_const)
#print '[DEBUG] GEN video pred 1 shape : ', GEN_video_pred_1.shape, ' GEN video h 1 shape : ', GEN_video_h_1.shape
GEN_video_pred_1_last = GetLastRNNOutput(GEN_video_pred_1)
GEN_video_1 = tf.concat([GEN_video_0, GEN_video_pred_1_last], axis = 1)
#print '[DEBUG] GEN video 1 shape : ', GEN_video_1.shape
#
[GEN_video_pred_2, GEN_video_h_2] = Generator(GEN_video_1, GEN_h_video_const)
#print '[DEBUG] GEN video pred 2 shape : ', GEN_video_pred_2.shape, ' GEN video h 2 shape : ', GEN_video_h_2.shape
GEN_video_pred_2_last = GetLastRNNOutput(GEN_video_pred_2)
GEN_video_2 = tf.concat([GEN_video_1, GEN_video_pred_2_last], axis = 1)
#print '[DEBUG] GEN video 2 shape : ', GEN_video_2.shape
#
[GEN_video_pred_3, GEN_video_h_3] = Generator(GEN_video_2, GEN_h_video_const)
#print '[DEBUG] GEN video pred 3 shape : ', GEN_video_pred_3.shape, ' GEN video h 3 shape : ', GEN_video_h_3.shape
GEN_video_pred_3_last = GetLastRNNOutput(GEN_video_pred_3)
GEN_video_3 = tf.concat([GEN_video_2, GEN_video_pred_3_last], axis = 1)
#print '[DEBUG] GEN video 3 shape : ', GEN_video_3.shape
#
[GEN_video_pred_4, GEN_video_h_4] = Generator(GEN_video_3, GEN_h_video_const)
#print '[DEBUG] GEN video pred 4 shape : ', GEN_video_pred_4.shape, ' GEN video h 4 shape : ', GEN_video_h_4.shape
GEN_video_pred_4_last = GetLastRNNOutput(GEN_video_pred_4)
GEN_video_4 = tf.concat([GEN_video_3, GEN_video_pred_4_last], axis = 1)
#print '[DEBUG] GEN video 4 shape : ', GEN_video_4.shape
#
[GEN_video_pred_5, GEN_video_h_5] = Generator(GEN_video_4, GEN_h_video_const)
#print '[DEBUG] GEN video pred 5 shape : ', GEN_video_pred_5.shape, ' GEN video h 5 shape : ', GEN_video_h_5.shape
GEN_video_pred_5_last = GetLastRNNOutput(GEN_video_pred_5)
GEN_video_5 = tf.concat([GEN_video_4, GEN_video_pred_5_last], axis = 1)
#print '[DEBUG] GEN video 5 shape : ', GEN_video_5.shape
# Define predicted image as well!!
# It is nothing but GEN_video_pred_5_last
# The below chunk is EXTRA
# [GEN_video_pred_6, GEN_video_h_6] = Generator(GEN_video_5, GEN_h_video_const)
# print '[DEBUG] GEN video pred 6 shape : ', GEN_video_pred_6.shape, ' GEN video h 6 shape : ', GEN_video_h_6.shape
# GEN_video_pred_6_last = GetLastRNNOutput(GEN_video_pred_6)
# GEN_video_6 = tf.concat([GEN_video_5, GEN_video_pred_6_last], axis = 1)
# print '[DEBUG] GEN video 5 shape : ', GEN_video_6.shape
[DIS_GEN_video_pred_1, DIS_GEN_video_h_1] = Discriminator(GEN_video_pred_1, GEN_h_video_const)
DIS_GEN_video_pred_1_last = GetLastRNNOutput(DIS_GEN_video_pred_1)
DIS_GEN_video_pred_1_last_1 = tf.reshape(DIS_GEN_video_pred_1_last, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_video_pred_1 shape : ', DIS_GEN_video_pred_1.shape, ' DIS_GEN_video_h_1 shape : ', DIS_GEN_video_h_1.shape, ' DIS_GEN_video_pred_1_last shape : ', DIS_GEN_video_pred_1_last.shape
[DIS_GEN_video_pred_2, DIS_GEN_video_h_2] = Discriminator(GEN_video_pred_2, GEN_h_video_const)
DIS_GEN_video_pred_2_last = GetLastRNNOutput(DIS_GEN_video_pred_2)
DIS_GEN_video_pred_2_last_1 = tf.reshape(DIS_GEN_video_pred_2_last, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_video_pred_2 shape : ', DIS_GEN_video_pred_2.shape, ' DIS_GEN_video_h_2 shape : ', DIS_GEN_video_h_2.shape, ' DIS_GEN_video_pred_2_last shape : ', DIS_GEN_video_pred_2_last.shape 
[DIS_GEN_video_pred_3, DIS_GEN_video_h_3] = Discriminator(GEN_video_pred_3, GEN_h_video_const)
DIS_GEN_video_pred_3_last = GetLastRNNOutput(DIS_GEN_video_pred_3)
DIS_GEN_video_pred_3_last_1 = tf.reshape(DIS_GEN_video_pred_3_last, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_video_pred_3 shape : ', DIS_GEN_video_pred_3.shape, ' DIS_GEN_video_h_3 shape : ', DIS_GEN_video_h_3.shape, ' DIS_GEN_video_pred_3_last shape : ', DIS_GEN_video_pred_3_last.shape 
[DIS_GEN_video_pred_4, DIS_GEN_video_h_4] = Discriminator(GEN_video_pred_4, GEN_h_video_const)
DIS_GEN_video_pred_4_last = GetLastRNNOutput(DIS_GEN_video_pred_4)
DIS_GEN_video_pred_4_last_1 = tf.reshape(DIS_GEN_video_pred_4_last, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_video_pred_4 shape : ', DIS_GEN_video_pred_4.shape, ' DIS_GEN_video_h_4 shape : ', DIS_GEN_video_h_4.shape, ' DIS_GEN_video_pred_4_last shape : ', DIS_GEN_video_pred_4_last.shape
[DIS_GEN_video_pred_5, DIS_GEN_video_h_5] = Discriminator(GEN_video_pred_5, GEN_h_video_const)
DIS_GEN_video_pred_5_last = GetLastRNNOutput(DIS_GEN_video_pred_5)
DIS_GEN_video_pred_5_last_1 = tf.reshape(DIS_GEN_video_pred_5_last, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_video_pred_5 shape : ', DIS_GEN_video_pred_5.shape, ' DIS_GEN_video_h_5 shape : ', DIS_GEN_video_h_5.shape, ' DIS_GEN_video_pred_5_last shape : ', DIS_GEN_video_pred_5_last.shape


####################################################################################################
########################GROUND TRUTH CONTEXT ENC-DEC FORWARD PASS###################################
####################################################################################################


# This is the third type of RNN # Define the forward path through generator with WITH CONTEXT AS GROUD TRUTH. We are giving out more information for the Generator now
GEN_seq_0 = tf.truncated_normal(dtype = tf.float32, mean = 0.0, stddev = 0.5, shape = [BATCH_SIZE, 1, LATENT_DIM], name = 'GEN_seq_0') # Pass the initial noise here
GEN_h_seq_const = tf.zeros([BATCH_SIZE, CELL_SIZE*STACK_SIZE]) # This will be constant
GEN_total_seq = tf.concat([GEN_seq_0, encoded_vector], axis = 1)
#print '[DEBUG] ################################################################################'
#print '[DEBUG] ################################################################################'
#print '[DEBUG] GEN_total_seq shape : ', GEN_total_seq.shape
GEN_seq_part_0 = tf.slice(GEN_total_seq, [0, 0, 0], [BATCH_SIZE, 1, CELL_SIZE])
#print '[DEBUG] GEN_seq_part_0 shape : ', GEN_seq_part_0.shape
GEN_seq_part_1 = tf.slice(GEN_total_seq, [0, 0, 0], [BATCH_SIZE, 2, CELL_SIZE])
#print '[DEBUG] GEN_seq_part_1 shape : ', GEN_seq_part_1.shape
GEN_seq_part_2 = tf.slice(GEN_total_seq, [0, 0, 0], [BATCH_SIZE, 3, CELL_SIZE])
#print '[DEBUG] GEN_seq_part_2 shape : ', GEN_seq_part_2.shape
GEN_seq_part_3 = tf.slice(GEN_total_seq, [0, 0, 0], [BATCH_SIZE, 4, CELL_SIZE])
#print '[DEBUG] GEN_seq_part_3 shape : ', GEN_seq_part_3.shape
GEN_seq_part_4 = tf.slice(GEN_total_seq, [0, 0, 0], [BATCH_SIZE, 5, CELL_SIZE])
#print '[DEBUG] GEN_seq_part_4 shape : ', GEN_seq_part_4.shape
GEN_seq_part_5 = tf.slice(GEN_total_seq, [0, 0, 0], [BATCH_SIZE, 6, CELL_SIZE])
#print '[DEBUG] GEN_seq_part_5 shape : ', GEN_seq_part_5.shape
#
[GEN_seq_pred_1, GEN_seq_h_1] = Generator(GEN_seq_part_0, GEN_h_seq_const)
#print '[DEBUG] GEN_seq_pred_1 shape : ', GEN_seq_pred_1.shape
[GEN_seq_pred_2, GEN_seq_h_2] = Generator(GEN_seq_part_1, GEN_h_seq_const)
#print '[DEBUG] GEN_seq_pred_2 shape : ', GEN_seq_pred_2.shape
[GEN_seq_pred_3, GEN_seq_h_3] = Generator(GEN_seq_part_2, GEN_h_seq_const)
#print '[DEBUG] GEN_seq_pred_3 shape : ', GEN_seq_pred_3.shape
[GEN_seq_pred_4, GEN_seq_h_4] = Generator(GEN_seq_part_3, GEN_h_seq_const)
#print '[DEBUG] GEN_seq_pred_4 shape : ', GEN_seq_pred_4.shape
[GEN_seq_pred_5, GEN_seq_h_5] = Generator(GEN_seq_part_4, GEN_h_seq_const)
#print '[DEBUG] GEN_seq_pred_5 shape : ', GEN_seq_pred_5.shape
[GEN_seq_pred_6, GEN_seq_h_6] = Generator(GEN_seq_part_5, GEN_h_seq_const)
#print '[DEBUG] GEN_seq_pred_6 shape : ', GEN_seq_pred_6.shape
# Discriminator pass
[DIS_GEN_seq_pred_1, DIS_GEN_seq_h_1] = Discriminator(GEN_seq_pred_1, GEN_h_seq_const)
DIS_GEN_seq_pred_1_last = GetLastRNNOutput(DIS_GEN_seq_pred_1)
DIS_GEN_seq_pred_1_last_1 = tf.reshape(DIS_GEN_seq_pred_1_last, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_seq_pred_1 shape : ', DIS_GEN_seq_pred_1.shape, ' DIS_GEN_seq_h_1 shape : ', DIS_GEN_seq_h_1.shape, ' DIS_GEN_seq_pred_1_last shape : ', DIS_GEN_seq_pred_1_last.shape
[DIS_GEN_seq_pred_2, DIS_GEN_seq_h_2] = Discriminator(GEN_seq_pred_2, GEN_h_seq_const)
DIS_GEN_seq_pred_2_last = GetLastRNNOutput(DIS_GEN_seq_pred_2)
DIS_GEN_seq_pred_2_last_1 = tf.reshape(DIS_GEN_seq_pred_2_last, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_seq_pred_2 shape : ', DIS_GEN_seq_pred_2.shape, ' DIS_GEN_seq_h_2 shape : ', DIS_GEN_seq_h_2.shape, ' DIS_GEN_seq_pred_2_last shape : ', DIS_GEN_seq_pred_2_last.shape
[DIS_GEN_seq_pred_3, DIS_GEN_seq_h_3] = Discriminator(GEN_seq_pred_3, GEN_h_seq_const)
DIS_GEN_seq_pred_3_last = GetLastRNNOutput(DIS_GEN_seq_pred_3)
DIS_GEN_seq_pred_3_last_1 = tf.reshape(DIS_GEN_seq_pred_3_last, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_seq_pred_3 shape : ', DIS_GEN_seq_pred_3.shape, ' DIS_GEN_seq_h_3 shape : ', DIS_GEN_seq_h_3.shape, ' DIS_GEN_seq_pred_3_last shape : ', DIS_GEN_seq_pred_3_last.shape
[DIS_GEN_seq_pred_4, DIS_GEN_seq_h_4] = Discriminator(GEN_seq_pred_4, GEN_h_seq_const)
DIS_GEN_seq_pred_4_last = GetLastRNNOutput(DIS_GEN_seq_pred_4)
DIS_GEN_seq_pred_4_last_1 = tf.reshape(DIS_GEN_seq_pred_4_last, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_seq_pred_4 shape : ', DIS_GEN_seq_pred_4.shape, ' DIS_GEN_seq_h_4 shape : ', DIS_GEN_seq_h_4.shape, ' DIS_GEN_seq_pred_4_last shape : ', DIS_GEN_seq_pred_4_last.shape
[DIS_GEN_seq_pred_5, DIS_GEN_seq_h_5] = Discriminator(GEN_seq_pred_5, GEN_h_seq_const)
DIS_GEN_seq_pred_5_last = GetLastRNNOutput(DIS_GEN_seq_pred_5)
DIS_GEN_seq_pred_5_last_1 = tf.reshape(DIS_GEN_seq_pred_5_last, [-1, CELL_SIZE])
#print '[DEBUG] DIS_GEN_seq_pred_5 shape : ', DIS_GEN_seq_pred_5.shape, ' DIS_GEN_seq_h_5 shape : ', DIS_GEN_seq_h_5.shape, ' DIS_GEN_seq_pred_5_last shape : ', DIS_GEN_seq_pred_5_last.shape


####################################################################################################
####################################DIS OF REAL DATA################################################
####################################################################################################


# Define the loss with corresponding entries
GEN_true_1 = tf.slice(encoded_vector, [0, 0, 0], [BATCH_SIZE, 1, CELL_SIZE])
#print '[DEBUG] GEN true 1 shape ', GEN_true_1.shape
GEN_true_2 = tf.slice(encoded_vector, [0, 0, 0], [BATCH_SIZE, 2, CELL_SIZE])
#print '[DEBUG] GEN true 2 shape ', GEN_true_2.shape
GEN_true_3 = tf.slice(encoded_vector, [0, 0, 0], [BATCH_SIZE, 3, CELL_SIZE])
#print '[DEBUG] GEN true 3 shape ', GEN_true_3.shape
GEN_true_4 = tf.slice(encoded_vector, [0, 0, 0], [BATCH_SIZE, 4, CELL_SIZE])
#print '[DEBUG] GEN true 4 shape ', GEN_true_4.shape
GEN_true_5 = tf.slice(encoded_vector, [0, 0, 0], [BATCH_SIZE, 5, CELL_SIZE])
#print '[DEBUG] GEN true 5 shape ', GEN_true_5.shape
# Define discriminator
[DIS_GEN_pred_true_1, DIS_GEN_true_h_1] = Discriminator(GEN_true_1, GEN_h_0)
DIS_GEN_pred_true_1_last = GetLastRNNOutput(DIS_GEN_pred_true_1)
DIS_GEN_pred_true_1_last_1 = tf.reshape(DIS_GEN_pred_true_1_last, [-1, CELL_SIZE])
#
[DIS_GEN_pred_true_2, DIS_GEN_true_h_2] = Discriminator(GEN_true_2, GEN_h_0)
DIS_GEN_pred_true_2_last = GetLastRNNOutput(DIS_GEN_pred_true_2)
DIS_GEN_pred_true_2_last_1 = tf.reshape(DIS_GEN_pred_true_2_last, [-1, CELL_SIZE])
#
[DIS_GEN_pred_true_3, DIS_GEN_true_h_3] = Discriminator(GEN_true_3, GEN_h_0)
DIS_GEN_pred_true_3_last = GetLastRNNOutput(DIS_GEN_pred_true_3)
DIS_GEN_pred_true_3_last_1 = tf.reshape(DIS_GEN_pred_true_3_last, [-1, CELL_SIZE])
#
[DIS_GEN_pred_true_4, DIS_GEN_true_h_4] = Discriminator(GEN_true_4, GEN_h_0)
DIS_GEN_pred_true_4_last = GetLastRNNOutput(DIS_GEN_pred_true_4)
DIS_GEN_pred_true_4_last_1 = tf.reshape(DIS_GEN_pred_true_4_last, [-1, CELL_SIZE])
#
[DIS_GEN_pred_true_5, DIS_GEN_true_h_5] = Discriminator(GEN_true_5, GEN_h_0)
DIS_GEN_pred_true_5_last = GetLastRNNOutput(DIS_GEN_pred_true_5)
DIS_GEN_pred_true_5_last_1 = tf.reshape(DIS_GEN_pred_true_5_last, [-1, CELL_SIZE])


####################################################################################################
##################################DIS AND GEN LOSS DEF##############################################
####################################################################################################


# Evaluate the logits by DIS_W_dec and DIS_B_dec
# REAL
DIS_of_REAL_1 = tf.add(tf.matmul(DIS_GEN_pred_true_1_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_REAL_2 = tf.add(tf.matmul(DIS_GEN_pred_true_2_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_REAL_3 = tf.add(tf.matmul(DIS_GEN_pred_true_3_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_REAL_4 = tf.add(tf.matmul(DIS_GEN_pred_true_4_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_REAL_5 = tf.add(tf.matmul(DIS_GEN_pred_true_5_last_1, DIS_W_dec), DIS_B_dec)
# VIDEO WITH SELF CONTEXT
DIS_of_FAKE_1_vid = tf.add(tf.matmul(DIS_GEN_video_pred_1_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_FAKE_2_vid = tf.add(tf.matmul(DIS_GEN_video_pred_2_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_FAKE_3_vid = tf.add(tf.matmul(DIS_GEN_video_pred_3_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_FAKE_4_vid = tf.add(tf.matmul(DIS_GEN_video_pred_4_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_FAKE_5_vid = tf.add(tf.matmul(DIS_GEN_video_pred_5_last_1, DIS_W_dec), DIS_B_dec)
# SEQ WITH GROUND CONTEXT
DIS_of_FAKE_1_seq = tf.add(tf.matmul(DIS_GEN_seq_pred_1_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_FAKE_2_seq = tf.add(tf.matmul(DIS_GEN_seq_pred_2_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_FAKE_3_seq = tf.add(tf.matmul(DIS_GEN_seq_pred_3_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_FAKE_4_seq = tf.add(tf.matmul(DIS_GEN_seq_pred_4_last_1, DIS_W_dec), DIS_B_dec)
DIS_of_FAKE_5_seq = tf.add(tf.matmul(DIS_GEN_seq_pred_5_last_1, DIS_W_dec), DIS_B_dec)


# Soft label in range (0.8, 1.2) 
soft_1_label = tf.random_uniform(shape = [BATCH_SIZE, 1], minval = 0.8, maxval = 1.2)
soft_0_label = tf.random_uniform(shape = [BATCH_SIZE, 1], minval = 0.0, maxval = 0.3)


# Define the real label loss for discriminator
DIS_REAL_LOSS_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_REAL_1, labels = soft_1_label))
DIS_REAL_LOSS_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_REAL_2, labels = soft_1_label))
DIS_REAL_LOSS_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_REAL_3, labels = soft_1_label))
DIS_REAL_LOSS_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_REAL_4, labels = soft_1_label))
DIS_REAL_LOSS_5 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_REAL_5, labels = soft_1_label))
DIS_REAL_LOSS = tf.add(DIS_REAL_LOSS_1, tf.add(DIS_REAL_LOSS_2, tf.add(DIS_REAL_LOSS_3, tf.add(DIS_REAL_LOSS_4, DIS_REAL_LOSS_5))))
DIS_REAL_LOSS_ = tf.add(DIS_REAL_LOSS_1, tf.add(DIS_REAL_LOSS_2, tf.add(DIS_REAL_LOSS_3, DIS_REAL_LOSS_4)))
DIS_REAL_LOSS__ = tf.add(DIS_REAL_LOSS_1, tf.add(DIS_REAL_LOSS_2, DIS_REAL_LOSS_3))
DIS_REAL_LOSS___ = tf.add(DIS_REAL_LOSS_1, DIS_REAL_LOSS_2)
DIS_REAL_LOSS____ = DIS_REAL_LOSS_1


# Define the fake label loss for discriminator (video type)
DIS_video_FAKE_LOSS_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_1_vid, labels = soft_0_label))
DIS_video_FAKE_LOSS_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_2_vid, labels = soft_0_label))
DIS_video_FAKE_LOSS_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_3_vid, labels = soft_0_label))
DIS_video_FAKE_LOSS_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_4_vid, labels = soft_0_label))
DIS_video_FAKE_LOSS_5 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_5_vid, labels = soft_0_label))
DIS_video_FAKE_LOSS = tf.add(DIS_video_FAKE_LOSS_1, tf.add(DIS_video_FAKE_LOSS_2, tf.add(DIS_video_FAKE_LOSS_3, tf.add(DIS_video_FAKE_LOSS_4, DIS_video_FAKE_LOSS_5))))
DIS_video_FAKE_LOSS_ = tf.add(DIS_video_FAKE_LOSS_1, tf.add(DIS_video_FAKE_LOSS_2, tf.add(DIS_video_FAKE_LOSS_3, DIS_video_FAKE_LOSS_4)))
DIS_video_FAKE_LOSS__ = tf.add(DIS_video_FAKE_LOSS_1, tf.add(DIS_video_FAKE_LOSS_2, DIS_video_FAKE_LOSS_3))
DIS_video_FAKE_LOSS___ = tf.add(DIS_video_FAKE_LOSS_1, DIS_video_FAKE_LOSS_2)
DIS_video_FAKE_LOSS____ = DIS_video_FAKE_LOSS_1


# Define the fake label loss for discriminator (seq type)
DIS_seq_FAKE_LOSS_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_1_seq, labels = soft_0_label))
DIS_seq_FAKE_LOSS_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_2_seq, labels = soft_0_label))
DIS_seq_FAKE_LOSS_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_3_seq, labels = soft_0_label))
DIS_seq_FAKE_LOSS_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_4_seq, labels = soft_0_label))
DIS_seq_FAKE_LOSS_5 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_5_seq, labels = soft_0_label))
DIS_seq_FAKE_LOSS = tf.add(DIS_seq_FAKE_LOSS_1, tf.add(DIS_seq_FAKE_LOSS_2, tf.add(DIS_seq_FAKE_LOSS_3, tf.add(DIS_seq_FAKE_LOSS_4, DIS_seq_FAKE_LOSS_5))))
DIS_seq_FAKE_LOSS_ = tf.add(DIS_seq_FAKE_LOSS_1, tf.add(DIS_seq_FAKE_LOSS_2, tf.add(DIS_seq_FAKE_LOSS_3, DIS_seq_FAKE_LOSS_4)))
DIS_seq_FAKE_LOSS__ = tf.add(DIS_seq_FAKE_LOSS_1, tf.add(DIS_seq_FAKE_LOSS_2, DIS_seq_FAKE_LOSS_3))
DIS_seq_FAKE_LOSS___ = tf.add(DIS_seq_FAKE_LOSS_1, DIS_seq_FAKE_LOSS_2)
DIS_seq_FAKE_LOSS____ = DIS_seq_FAKE_LOSS_1



# Define the fake label loss for generator (video type)
GEN_video_FAKE_LOSS_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_1_vid, labels = soft_1_label))
GEN_video_FAKE_LOSS_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_2_vid, labels = soft_1_label))
GEN_video_FAKE_LOSS_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_3_vid, labels = soft_1_label))
GEN_video_FAKE_LOSS_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_4_vid, labels = soft_1_label))
GEN_video_FAKE_LOSS_5 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_5_vid, labels = soft_1_label))
GEN_video_FAKE_LOSS = tf.add(GEN_video_FAKE_LOSS_1, tf.add(GEN_video_FAKE_LOSS_2, tf.add(GEN_video_FAKE_LOSS_3, tf.add(GEN_video_FAKE_LOSS_4, GEN_video_FAKE_LOSS_5))))
GEN_video_FAKE_LOSS_ = tf.add(GEN_video_FAKE_LOSS_1, tf.add(GEN_video_FAKE_LOSS_2, tf.add(GEN_video_FAKE_LOSS_3, GEN_video_FAKE_LOSS_4)))
GEN_video_FAKE_LOSS__ = tf.add(GEN_video_FAKE_LOSS_1, tf.add(GEN_video_FAKE_LOSS_2, GEN_video_FAKE_LOSS_3))
GEN_video_FAKE_LOSS___ = tf.add(GEN_video_FAKE_LOSS_1, GEN_video_FAKE_LOSS_2)
GEN_video_FAKE_LOSS____ = GEN_video_FAKE_LOSS_1


# Define the fake label loss for generator (seq type)
GEN_seq_FAKE_LOSS_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_1_seq, labels = soft_1_label))
GEN_seq_FAKE_LOSS_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_2_seq, labels = soft_1_label))
GEN_seq_FAKE_LOSS_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_3_seq, labels = soft_1_label))
GEN_seq_FAKE_LOSS_4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_4_seq, labels = soft_1_label))
GEN_seq_FAKE_LOSS_5 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = DIS_of_FAKE_5_seq, labels = soft_1_label))
GEN_seq_FAKE_LOSS = tf.add(GEN_seq_FAKE_LOSS_1, tf.add(GEN_seq_FAKE_LOSS_2, tf.add(GEN_seq_FAKE_LOSS_3, tf.add(GEN_seq_FAKE_LOSS_4, GEN_seq_FAKE_LOSS_5))))
GEN_seq_FAKE_LOSS_ = tf.add(GEN_seq_FAKE_LOSS_1, tf.add(GEN_seq_FAKE_LOSS_2, tf.add(GEN_seq_FAKE_LOSS_3, GEN_seq_FAKE_LOSS_4)))
GEN_seq_FAKE_LOSS__ = tf.add(GEN_seq_FAKE_LOSS_1, tf.add(GEN_seq_FAKE_LOSS_2, GEN_seq_FAKE_LOSS_3))
GEN_seq_FAKE_LOSS___ = tf.add(GEN_seq_FAKE_LOSS_1, GEN_seq_FAKE_LOSS_2)
GEN_seq_FAKE_LOSS____ = GEN_seq_FAKE_LOSS_1


# DIS loss
DIS_LOSS = tf.add(DIS_REAL_LOSS, DIS_video_FAKE_LOSS)
DIS_LOSS_ = tf.add(DIS_REAL_LOSS_, DIS_video_FAKE_LOSS_)
DIS_LOSS__ = tf.add(DIS_REAL_LOSS__, DIS_video_FAKE_LOSS__)
DIS_LOSS___ = tf.add(DIS_REAL_LOSS___, DIS_video_FAKE_LOSS___)
DIS_LOSS____ = tf.add(DIS_REAL_LOSS____, DIS_video_FAKE_LOSS____)
#DIS_LOSS = DIS_REAL_LOSS + DIS_seq_FAKE_LOSS
GEN_LOSS = GEN_video_FAKE_LOSS
GEN_LOSS_ = GEN_video_FAKE_LOSS_
GEN_LOSS__ = GEN_video_FAKE_LOSS__
GEN_LOSS___ = GEN_video_FAKE_LOSS___
GEN_LOSS____ = GEN_video_FAKE_LOSS____
#GEN_LOSS = GEN_seq_FAKE_LOSS


# Optimizers
optimize_DIS_REAL_LOSS = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_REAL_LOSS)
optimize_DIS_REAL_LOSS_ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_REAL_LOSS_)
optimize_DIS_REAL_LOSS__ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_REAL_LOSS__)
optimize_DIS_REAL_LOSS___ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_REAL_LOSS___)
optimize_DIS_REAL_LOSS____ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_REAL_LOSS____)
#
optimize_DIS_FAKE_LOSS = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_video_FAKE_LOSS)
optimize_DIS_FAKE_LOSS_ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_video_FAKE_LOSS_)
optimize_DIS_FAKE_LOSS__ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_video_FAKE_LOSS__)
optimize_DIS_FAKE_LOSS___ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_video_FAKE_LOSS___)
optimize_DIS_FAKE_LOSS____ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_video_FAKE_LOSS____)
#
optimize_DIS_LOSS = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_LOSS)
optimize_DIS_LOSS_ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_LOSS_)
optimize_DIS_LOSS__ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_LOSS__)
optimize_DIS_LOSS___ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_LOSS___)
optimize_DIS_LOSS____ = tf.train.GradientDescentOptimizer(lr_GradientDescentOptimizer).minimize(DIS_LOSS____)
#
optimize_GEN_LOSS = tf.train.AdamOptimizer(lr_AdamOptimizer).minimize(GEN_LOSS)
optimize_GEN_LOSS_ = tf.train.AdamOptimizer(lr_AdamOptimizer).minimize(GEN_LOSS_)
optimize_GEN_LOSS__ = tf.train.AdamOptimizer(lr_AdamOptimizer).minimize(GEN_LOSS__)
optimize_GEN_LOSS___ = tf.train.AdamOptimizer(lr_AdamOptimizer).minimize(GEN_LOSS___)
optimize_GEN_LOSS____ = tf.train.AdamOptimizer(lr_AdamOptimizer).minimize(GEN_LOSS____)


####################################################################################################
######################################TRAINING LOOPS################################################
####################################################################################################


# Define the session and saver state
sess = tf.Session()
saver = tf.train.Saver()


# Initializer
init = tf.global_variables_initializer()
sess.run(init)


# Variables lists
trainable_variables_list = tf.trainable_variables()
print('[INFO] Trainable variables list : ' + ' Variables count : ' +  str(len(trainable_variables_list)))
for item in trainable_variables_list :
	print('[INFO] 		' +  str(item.name))
#
encoder_variables_list = [item for item in trainable_variables_list if 'ENC' in item.name ]
print('[INFO] Encoder variables list : ' + ' Variables count : ' + str(len(encoder_variables_list)))
for item in encoder_variables_list :
	print('[INFO] 		' + str(item.name))
#
decoder_variables_list = [item for item in trainable_variables_list if 'DEC' in item.name ]
print('[INFO] Decoder variables list : ' + ' Variables count : ' + str(len(decoder_variables_list)))
for item in decoder_variables_list :
	print('[INFO] 		' + str(item.name))
#
sampler_variables_list = [item for item in trainable_variables_list if 'SAM' in item.name ]
print('[INFO] Sampler variables list : ' + ' Variables count : ' + str(len(sampler_variables_list)))
for item in sampler_variables_list :
	print('[INFO] 		' + str(item.name))
#
auxiliary_variables_list = [item for item in trainable_variables_list if 'AUX' in item.name ]
print('[INFO] Auxiliary variables list : ' + ' Variables count : ', str(len(auxiliary_variables_list)))
for item in auxiliary_variables_list :
	print('[INFO] 		', str(item.name))
#
generator_variables_list = [item for item in trainable_variables_list if 'GEN' in item.name ]
print('[INFO] Generator variables list : ' + ' Variables count : ' + str(len(generator_variables_list)))
for item in generator_variables_list :
	print('[INFO] 		' + str(item.name))
#
discriminator_variables_list = [item for item in trainable_variables_list if 'DIS' in item.name ]
print('[INFO] Discriminator variables list : ' + ' Variables count : ' + str(len(discriminator_variables_list)))
for item in discriminator_variables_list :
	print('[INFO] 		' + str(item.name))
#
# print '[DEBUG] Terminating : '
# sys.exit()


# Training Loops
for iteration in range(ITR):
	#print '[DEBUG] Account for null definition error message : STUPID PYTHON'
	# Get next batch
	train_batch = data_loader.GetNextBatch()
	# Print iteration
	print('[TRAINING] Iteration : ' + str(iteration))
	# Train the VAE
	[vae_loss, _] = sess.run([variational_lower_bound_loss, optimize_VAE_Loss], feed_dict = {X_im : train_batch})
	if iteration%100 == 0 :
		print('[TRAINING] Iteration : ' + str(iteration) + 'Variational Auto Encoder Loss : ' + str(vae_loss))
	# Train the GAN in Sir's fashion
	[dis_loss, _] = sess.run([DIS_LOSS, optimize_DIS_LOSS], feed_dict = {X_im : train_batch})
	if iteration%100 == 0 :
		print('[TRAINING] Iteration : ' + str(iteration) + 'Discriminator Loss : ' + str(dis_loss))
	[gen_loss, _] = sess.run([GEN_LOSS, optimize_GEN_LOSS], feed_dict = {X_im : train_batch})
	if iteration%100 == 0 :
		print('[TRAINING] Iteration : ' + str(iteration) + 'Generator Loss : ' + str(gen_loss))
	#
	if iteration%5 == 0:
		data_it = []
		# a = sess.run([reconstr_1, reconstr_2, reconstr_3, reconstr_4, reconstr_5, reconstr_6])
		# print '[DEBUG] The reconstruction output : ', a 
		# print '[DEBUG] The reconstruction output length : ', len(a) # 6
		# print '[DEBUG] The reconstruction output entry shape: ', a[0].shape # (32, 128, 128, 1)
		[im1, im2, im3, im4, im5, im6] = sess.run([reconstr_1, reconstr_2, reconstr_3, reconstr_4, reconstr_5, reconstr_6])
		#im1_, im2_, im3_, im4_, im5_, im6_ = sess.run([reconstr_1, reconstr_2, reconstr_3, reconstr_4, reconstr_5, reconstr_6])
		# print '[DEBUG] im1 shape : ', im1.shape
		# print '[DEBUG] im2 shape : ', im2.shape
		# print '[DEBUG] im3 shape : ', im3.shape
		# print '[DEBUG] im4 shape : ', im4.shape
		# print '[DEBUG] im5 shape : ', im5.shape
		# print '[DEBUG] im6 shape : ', im6.shape
		# print '[DEBUG] im1_ shape : ', im1_.shape
		# print '[DEBUG] im2_ shape : ', im2_.shape
		# print '[DEBUG] im3_ shape : ', im3_.shape
		# print '[DEBUG] im4_ shape : ', im4_.shape
		# print '[DEBUG] im5_ shape : ', im5_.shape
		# print '[DEBUG] im6_ shape : ', im6_.shape
		# print '[DEBUG] Exiting : '
		# sys.exit()
		# Save the data generated
		data_it.append(im1)
		data_it.append(im2)
		data_it.append(im3)
		data_it.append(im4)
		data_it.append(im5)
		data_it.append(im6)
		data_it_np = np.array(data_it)
		np.save('Generated_Dataset/Generated_Images_' + str(iteration), data_it_np)
		# Save the states in models
		os.system('mkdir Models/Iteration_' + str(iteration))
		saver.save(sess, 'Models/Iteration_' + str(iteration) + '/Intermediate_Model_' + str(iteration))


####################################################################################################
##########################################EXIT######################################################
####################################################################################################
