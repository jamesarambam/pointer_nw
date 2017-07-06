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

# ============================================================================ #

# --------------------- Variables ------------------------------ #

ppath = os.getcwd() + "/"  # Project Path Location
BATCH_SIZE = 3
MAX_STEP = 2  # Number of numbers to sort.
RNN_SIZE = 2
INPUT_SIZE = 1
LEARNING_RATE = 1e-2


encoder_inputs = []
decoder_inputs = []
decoder_targets = []
target_weights = []



ax.clearDir("./logs")
# -------------------------------------------------------------- #

print "# ============================ START ============================ #"

def main():

    dataset = DataGenerator()
    cell = tf.contrib.rnn.GRUCell(RNN_SIZE)

    for i in range(MAX_STEP):
        encoder_inputs.append(tf.placeholder( tf.float32, [BATCH_SIZE, INPUT_SIZE], name="EncoderInput%d" % i))
    for i in range(MAX_STEP + 1):
        decoder_inputs.append(tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_SIZE], name="DecoderInput%d" % i))
        decoder_targets.append(tf.placeholder( tf.float32, [BATCH_SIZE, MAX_STEP + 1], name="DecoderTarget%d" % i))  # one hot
        target_weights.append(tf.placeholder(tf.float32, [BATCH_SIZE, 1], name="TargetWeight%d" % i))


    encoder_outputs, final_state = tf.contrib.rnn.static_rnn(cell, encoder_inputs, dtype=tf.float32)


    # Need a dummy output to point on it. End of decoding.
    encoder_outputs = [tf.zeros([BATCH_SIZE, RNN_SIZE])] + encoder_outputs


    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [tf.reshape(e, [-1, 1, cell.output_size])for e in encoder_outputs]
    attention_states = tf.concat(axis=1, values=top_states)


    with tf.variable_scope("decoder"):
        outputs, states, _ = pointer_decoder(decoder_inputs, final_state, attention_states, cell)


    with tf.name_scope('trainLoss'):
        trainLoss = tf.placeholder(tf.float32)
        tf.summary.scalar('trainLoss', trainLoss)


    loss = 0.0
    for op, tg, w in zip(outputs, decoder_targets, target_weights):
        loss += tf.nn.softmax_cross_entropy_with_logits(logits=op, labels=tg) * w


    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    train_loss_value = 0.0


    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000000):
            encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(BATCH_SIZE, MAX_STEP)

            print encoder_input_data
            exit()


            feed_dict = create_feed_dict(encoder_input_data, decoder_input_data, targets_data)
            d_x, l = sess.run([loss, train_op], feed_dict=feed_dict)
            train_loss_value = 0.9 * train_loss_value + 0.1 * d_x
            if i % 100 == 0:
                print('Step: %d' % i)
                print("Train: ", train_loss_value)
                summary = sess.run(merged, {trainLoss:train_loss_value})
                writer.add_summary(summary, i)



    # encoder_input_data, decoder_input_data, targets_data = dataset.next_batch(1, 2)
    # print (encoder_input_data)
    # print(decoder_input_data)
    # print (targets_data)


def create_feed_dict(encoder_input_data, decoder_input_data, decoder_target_data):
    feed_dict = {}
    for placeholder, data in zip(encoder_inputs, encoder_input_data):
        feed_dict[placeholder] = data

    for placeholder, data in zip(decoder_inputs, decoder_input_data):
        feed_dict[placeholder] = data

    for placeholder, data in zip(decoder_targets, decoder_target_data):
        feed_dict[placeholder] = data

    for placeholder in target_weights:
        feed_dict[placeholder] = np.ones([BATCH_SIZE, 1])

    return feed_dict




# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"
