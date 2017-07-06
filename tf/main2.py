"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 14 Jun 2017
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# ================================ priImports ================================ #

import sys
import os
import platform
from pprint import pprint
import time
import dataset
from dataset import DataGenerator
import tensorflow as tf
from pointer import pointer_decoder
import numpy as np

# ================================ secImports ================================ #

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



print "# ============================ START ============================ #"

# ============================================================================ #



# --------------------- Variables ------------------------------ #

# ------------------- Variables ------------------ #
NUM_BATCHES = 3
BATCH_SIZE = 2
MAX_STEP = 5  # Number of numbers to sort. Maximum Input Sequence Length
RNN_SIZE = 2  # Number of units in each layer of the model
INPUT_SIZE = 1 # Size of Input data
LEARNING_RATE = 1e-2
encoder_inputs = []
decoder_inputs = []
decoder_targets = []
target_weights = []



# -------------------------------------------------------------- #

def encoder():



def main():

    # ------------------ Data ----------------- #
    dataset = DataGenerator()
    for i in range(MAX_STEP):
        encoder_inputs.append(tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_SIZE], name="EncoderInput%d" % i))
    for i in range(MAX_STEP + 1):
        decoder_inputs.append(tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_SIZE], name="DecoderInput%d" % i))
        decoder_targets.append(
            tf.placeholder(tf.float32, [BATCH_SIZE, MAX_STEP + 1], name="DecoderTarget%d" % i))  # one hot
        target_weights.append(tf.placeholder(tf.float32, [BATCH_SIZE, 1], name="TargetWeight%d" % i))
    # ------------ Batches --------- #
    data = []
    for i in range(NUM_BATCHES):
        encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(BATCH_SIZE, MAX_STEP)
        data.append([encoder_input_data, decoder_input_data, targets_data])

    # --------------- Cell ---------------- #
    cell = tf.contrib.rnn.GRUCell(RNN_SIZE)
    # ------- Encoder RNN ------------ #
    encoder_outputs, final_state = tf.contrib.rnn.static_rnn(cell, encoder_inputs, dtype=tf.float32)






    exit()
    # -------------- tf.Sessions -------------- #
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        ax.clearDir("./logs")
        writer = tf.summary.FileWriter("./logs/", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        for encoder_input_data, decoder_input_data, targets_data in data:
            feed_dict = {}
            for placeholder, data in zip(encoder_inputs, encoder_input_data):
                feed_dict[placeholder] = data
            en = sess.run([], feed_dict=feed_dict)




            # =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"