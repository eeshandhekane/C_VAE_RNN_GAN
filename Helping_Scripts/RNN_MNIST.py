# Dependencies
import tensorflow as tf
import numpy as np
import os, sys, re


# Extract MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)


# Parameters
BATCH_SIZE = 128
CELL_SIZE = 28
STACK_SIZE = 2
UNFOLD_SIZE = 28 # This is the SEQ_LEN param in google talks
NUM_CLASS = 10
ITR = 10000


# Define placeholders and variables
X = tf.placeholder(tf.float32, [None, 784])
H_in = tf.placeholder(tf.float32, [None, CELL_SIZE * STACK_SIZE])
# Define classifier weights
W = tf.Variable(tf.truncated_normal([CELL_SIZE*UNFOLD_SIZE, NUM_CLASS], stddev = 0.1))
B = tf.Variable(tf.ones([NUM_CLASS])/10)
Y_true = tf.placeholder(tf.float32, [None, NUM_CLASS])


# Define the RNN/LSTM/GRU cell
gru_cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
multi_gru_cell = tf.contrib.rnn.MultiRNNCell([gru_cell]*STACK_SIZE, state_is_tuple = False)
X_ = tf.reshape(X, [BATCH_SIZE, UNFOLD_SIZE, CELL_SIZE])
H_r, H = tf.nn.dynamic_rnn(multi_gru_cell, X_, initial_state = H_in)
# H_r has the list of all the hidden states: [BATCH_SIZE, UNFOLD_SIZE, CELL_SIZE] for each entry in batch, for each time instant, there is an output of cell size 
# We device the classification task as using ALL THE INTERMEDIATE STATES in H_r to collectively predict the class of the image
H_r_essence = tf.reshape(H_r, [BATCH_SIZE, -1]) # We collect "all the data generated on the go" in the hidden states
Y_logits = tf.add(tf.matmul(H_r_essence, W), B) # The output of this fully connected layer are defined as the logits
Y_pred = tf.nn.softmax(Y_logits)
class_pred = tf.argmax(Y_pred, 1)
class_true = tf.argmax(Y_true, 1)
corr_pred = tf.equal(class_pred, class_true)
corr_perc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits(logits = Y_logits, labels = Y_true)
optim = tf.train.AdamOptimizer(1e-3)
training_step = optim.minimize(softmax_cross_entropy_with_logits)


# Define a session
sess = tf.Session()


# Initialize all variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)


# # Define the testing batch
# test_batch_X = mnist.test.next_batch(BATCH_SIZE)
# test_batch_Y = mnist.test.next_batch(BATCH_SIZE)
# print '[DEBUG] ', test_batch_X.shape
# print '[DEBUG] ', test_batch_Y.shape
# THERE IS SOME GLITCH WITH mnist.test.next_batch


# Training loop
for itr in range(ITR):
	# Get the training data
	batch_X, batch_Y = mnist.train.next_batch(BATCH_SIZE)
	# There is a glitch with the mnist.test.next_batch thing. Is there a problem with mnist.train.next_batch??
	# print '[DEBUG] ', batch_X.shape
	# print '[DEBUG] ', batch_Y.shape
	# # NO, THERE ISN'T!!! This may be causing problems in Tanaya Mam's Code??
	# Define the input state
	H_init = np.zeros([BATCH_SIZE, CELL_SIZE*STACK_SIZE])
	# Define training batch
	feed_dict = { X : batch_X , Y_true : batch_Y , H_in : H_init }
	# Train
	sess.run([training_step], feed_dict = feed_dict)
	# Test the performance ocassionally
	if itr%100 == 0 :
		# Define testing batch as the training batch itself
		c_pred, c_true, corr_acc = sess.run([class_true, class_pred, corr_perc], feed_dict = feed_dict) # Never match the names in return values and actual global variables
		# Display
		print '[TESTING] Iteration : ', itr
		print '[TESTING] 	Predicted classes : ', c_pred
		print '[TESTING] 	True classes : ', c_true 
		print '[TESTING] 	Accuracy : ', corr_acc


# This gets us to the performance of 1.0 as well! Well done!!