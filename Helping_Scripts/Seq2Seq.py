# Dependencies
import cv2
import tensorflow as tf
import numpy as np
import helpers


# This architecture is bidirectional encoder and decoder architecture
# Learns to output a random length sequence from a random length input sequence


# Clean the graph!! Custom
tf.reset_default_graph()


# Define the session
sess = tf.Session()


# Check version
print '[INFO] TF Version : ', tf.__version__


# We need to have fixed sized vectors as inputs to RNN, but we need to model 
# Thus, we need to perform padding
# Also, we need to define end-of-data.
# For these purposes, there needs to be a token


# Parameters
PAD = 0 # Padding value
EOS = 1 # End-of-Sentence value
VOCAB_SIZE = 10 # The size of words
INPUT_EMBED_SIZE = 20 # We need to convert the input into data of size 20 which can be fed into an RNN
ENCODER_HIDDEN_UNITS = 20
DECODER_HIDDEN_UNITS = 2*ENCODER_HIDDEN_UNITS # Usually, 2 in this equation should be 1


# Define placeholders and variables
encoder_input = tf.placeholder(shape = [None, None], dtype = tf.int32, name = 'encoder_input')
encoder_input_length = tf.placeholder(shape = [None,], dtype = tf.int32, name = 'encoder_input_length')
decoder_target = tf.placeholder(shape = [None, None], dtype = tf.int32, name = 'decoder_output')


# Embedding
embedding = tf.Variable(tf.random_uniform((VOCAB_SIZE, INPUT_EMBED_SIZE), -1.0, 1.0), dtype = tf.float32) # From -1 to 1, generate random numbers
encoder_input_embedded = tf.nn.embedding_lookup(embedding, encoder_input)


# Import encoder cells and LSTMStateTuple
from tensorflow.contrib.rnn import LSTMCell, GRUCell, LSTMStateTuple


# Define encoder cell
encoder_cell = LSTMCell(ENCODER_HIDDEN_UNITS)


# Define the Seq2Seq model
#(encoder_fw_output, encoder_bw_output), (encoder_fw_final_state, encoder_bw_final_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, cell_bw = encoder_cell, inputs = encoder_input_embedded, sequence_length = encoder_input_length, dtype = tf.float64, time_major = True) # use tf.float32
(encoder_fw_output, encoder_bw_output), (encoder_fw_final_state, encoder_bw_final_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, cell_bw = encoder_cell, inputs = encoder_input_embedded, sequence_length = encoder_input_length, dtype = tf.float32, time_major = True)


# Define the outputs
encoder_output = tf.concat([encoder_fw_output, encoder_bw_output], 2)
encoder_final_state_c = tf.concat([encoder_fw_final_state.c, encoder_bw_final_state.c], 1) # Output value!!
encoder_final_state_h = tf.concat([encoder_fw_final_state.h, encoder_bw_final_state.h], 1) # Hidden State!!
encoder_final_state = LSTMStateTuple(c = encoder_final_state_c, h = encoder_final_state_h) # This thing is going to be fed into the decoder!!


# Define decoder cell
decoder_cell = LSTMCell(DECODER_HIDDEN_UNITS)
encoder_max_time, BATCH_SIZE = tf.unstack(tf.shape(encoder_input))
#print '[INFO] BATCH_SIZE : ', BATCH_SIZE
# +3 for 2 additional steps and EOS token at the end
decoder_length = encoder_input_length + 3
# There are no models where the model knows if the sentence is over or not


# We are planning to model attention ourselves
W = tf.Variable(tf.random_uniform([DECODER_HIDDEN_UNITS, VOCAB_SIZE], -1, 1), dtype = tf.float32)
B = tf.Variable(tf.zeros([VOCAB_SIZE]), dtype = tf.float32)


# Assert true and generate an error otherwise
assert EOS == 1 and PAD == 0


# Define EOS and PAD chunks and define look-ups
eos_time_slice = tf.ones([BATCH_SIZE], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([BATCH_SIZE], dtype=tf.int32, name='PAD')
eos_step_embedded = tf.nn.embedding_lookup(embedding, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embedding, pad_time_slice)	


# We code decoder manually to work
# Note that tf.nn.dynamic_rnn requires all inputs (t_0, ..., t_n) to be passed in advance.
# The dynamic part of its name comes from this behaviour to work with variable n as well
# But, if we want to perform some tasks wherein the outputs at instant t - 1 define what specific needs to be done in the time step t
# This is achieved by tf.nn.raw_rnn
# tf.nn.raw_rnn runs by defining "loop transition function", which defines inputs of state t using the output and state of t - 1
# A loop transition function inputs : (time, previous_cell_output, previous_cell_state, previous_loop_state) --> (elements_finished, input, cell_state, output, loop_state)
# At time 0, everything is "None", but afterwards, everything is a "Tensor"
# Initial call at t = 0 should provide initial state and inputs to RNN
# Transition call should define the transition from one step to another
# Loop function must take arguments and produce outputs as shown below. Their type, number and arguments is EXACTLY AS SHOWN in loop_function


# We define the initial call
def loop_initial():
	# Initial elements that are covered
	initial_elements_finished = (0 >= decoder_length)
	# Define the intial input (probabily, this is coming from the time where the encoder gives EOS)
	initial_input = eos_step_embedded
	# Define the initial state of the decoder (which IS the final state of the encoder)
	initial_cell_state = encoder_final_state
	# Define the initial cell output (nothing has happened as of now)
	initial_cell_output = None
	# Define the additional information to be passed in the loop (nothing as of now)
	initial_loop_state = None
	# Return the tuple. Note the order!!
	return (initial_elements_finished, initial_input, initial_cell_state, initial_cell_output, initial_loop_state)


# For this code, we need to understand how tf.nn.embedding_lookup works
# It inputs params as first input and lookup ids as the second
# It searches for the embedding in the input params by the id and returns it!!
# Similar to--
# If params = [[1, 2], [3, 4], [5, 6]] and ids = [0, 2, 0], then return value is [[1, 2], [5, 6], [1, 2]]


# Define the transition call
def loop_transition(time, previous_output, previous_state, previous_loop_state):
	# Define the function to get the next input 
	def get_next_input():
		# Define the logits obtained by matmul (logits as the features are not probability distributions and do not add up to 1)
		output_logits = tf.add(tf.matmul(previous_output, W), B)
		# Define the prediction as argmax of output_logits (no need of softmax, the leader reamins the leader!!)
		prediction = tf.argmax(output_logits, axis=1)	
		# This gives the next input
		next_input = tf.nn.embedding_lookup(embedding, prediction)
		return next_input
	# Define the finished elements
	elements_finished = (time >= decoder_length)
	# Take and of all the elements. tf.reduce_all does this. If we mention dimensions, it does by selecting elements across that dimension. Otherwise, it reduces for all entries
	finished = tf.reduce_all(elements_finished)
	# Conditional input using tf!! tf.cond inputs the pred value as finished and if True, evaluates "lambda..._embedded" and if False, evaluates "get_next_input"
	# Note that both the functions, true_fn and false_fn must return the same type of data type or output type and must have the same number of outputs in the same order
	input = tf.cond(finished, lambda : pad_step_embedded, get_next_input)
	# Define theh things to return!! Note the order
	state = previous_state
	output = previous_output
	loop_state = None
	# Return!!
	return (elements_finished, input, state, output, loop_state)


# Define the loop function
def loop_function(time, previous_output, previous_state, previous_loop_state):
	# Check if the previous time step is 0
	if previous_state is None :
		# Safe-check
		assert previous_output is None and previous_state is None
		# Return the function for initial call
		return loop_initial()
	# Check if the previous time step is NOT 0
	else:
		# Return the function for transition call
		return loop_transition(time, previous_output, previous_state, previous_loop_state)


# Now, we consider what is returned by tf.nn.raw_rnn
# It returns a tuple (emit_tensor_array, final_state, final_loop_state)
# The first argument is an array of tensors which are in the 4th argument (output) of return value of raw_rnn
# Thus, there will be as many tensors in emit_array_tensor as are the time steps unfolded


# Run the dynamic rnn
decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_function)
# Stack the output to get the tensor which is the decoder output
decoder_output = decoder_outputs_ta.stack()


# Unstack the output along a dimension to get smaller instances as slices
decoder_max_step, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_output))
# Flatten each decoder output
decoder_outputs_flat = tf.reshape(decoder_output, (-1, decoder_dim))
# Pass through decoder
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), B)
# Prediction values
decoder_logit = tf.reshape(decoder_logits_flat, (decoder_max_step, decoder_batch_size, VOCAB_SIZE))
# Predictions
decoder_prediction = tf.argmax(decoder_logit, 2)


# Loss and Optimizer
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(decoder_target, depth = VOCAB_SIZE, dtype = tf.float32), logits = decoder_logit)
loss = tf.reduce_mean(cross_entropy_loss)
training_step = tf.train.AdamOptimizer().minimize(cross_entropy_loss)


# Initialize all variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)


# Feed data
BATCH_SIZE = 100
# Get batch
batches = helpers.random_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10,batch_size=BATCH_SIZE)
# Print first few
print('[INFO] First few sequences : ')
for seq in next(batches)[:5]:
	print '[INFO] 	', seq


# Define a function for the next feed
def next_feed():
    batch = next(batches)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
    )
    return {
        encoder_input: encoder_inputs_,
        encoder_input_length: encoder_input_lengths_,
        decoder_target: decoder_targets_,
    }


# Train
loss_track = []
max_batches = 10000
batches_in_epoch = 1000

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([training_step, cross_entropy_loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_input].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()

except KeyboardInterrupt:
    print('training interrupted')
