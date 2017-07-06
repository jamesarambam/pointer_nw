"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 06 Jul 2017
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# ================================ secImports ================================ #

import sys
import os
import platform
from pprint import pprint
import time

o = platform.system()
if o == "Linux":
    d = platform.dist()
    if d[0] == "debian":
        sys.path.append("/home/james/Codes/python")
    if d[0] == "centos":
        sys.path.append("/home/arambamjs.2016/Codes/python")
if o == "Darwin":
    sys.path.append("/Users/james/Codes/python")

import auxlib.auxLib as ax

# ================================ priImports ================================ #
from keras import initializers
from keras.layers.recurrent import _time_distributed_dense
from keras.activations import tanh, softmax
from keras.layers import LSTM
from keras.engine import InputSpec
from parameters import LEARNING_RATE, NB_EPOCH, BATCH_SIZE, TEST_BATCH
import keras.backend as K
import numpy as np
# --------------------- Variables ------------------------------ #
ppath = os.getcwd() + "/"  # Project Path Location
# -------------------------------------------------------------- #

class pointerLayer(LSTM):
    def __init__(self, hidden_shape, *args, **kwargs):
        self.hidden_shape = hidden_shape
        super(pointerLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(pointerLayer, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        init = initializers.get('orthogonal')
        self.W1 = K.variable(init((self.hidden_shape, 1)))
        self.W2 = K.variable(init((self.hidden_shape, 1)))
        self.vt = K.variable(init((input_shape[1], 1)))
        self.trainable_weights += [self.W1, self.W2, self.vt]

    def call(self, x, mask=None):

        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1]-1, :]
        x_input = K.repeat(x_input, input_shape[1])
        initial_states = self.get_initial_state(x_input)
        constants = super(pointerLayer, self).get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])
        return outputs

    def step(self, x_input, states):

        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super(pointerLayer, self).step(x_input, states[:-1])
        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = _time_distributed_dense(en_seq, self.W1, output_dim=1)
        Dij = _time_distributed_dense(dec_seq, self.W2, output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)
        # ----------- Probability tensor ----------------- #
        pointer = softmax(U)
        return pointer, [h, c]

    def compute_output_shape(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])

def scheduler(epoch):
    return LEARNING_RATE
    if epoch < NB_EPOCH/4:
        return LEARNING_RATE
    elif epoch < NB_EPOCH/2:
        return LEARNING_RATE*0.5
    return LEARNING_RATE*0.1

def predict(X_test, Y_test, model):

    print "\n\n\n"
    print "--------------------------- Prediction --------------------------- "
    print "[X_test]", "[Y_test]", "[Y_Prediction]"
    predictions = model.predict(X_test)
    pred_index = np.array([np.argmax(predictions[i],0) for i in xrange(len(predictions))])
    for i in range(TEST_BATCH):
        print map(lambda v : round(v[0], 4), [list(Y_test[i][j]) for j in range(len(Y_test[i]))]), map(lambda v : round(v[0], 4), [list(Y_test[i][j]) for j in range(len(Y_test[i]))]), pred_index[i]
