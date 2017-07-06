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
from keras.models import Model
from keras.layers import LSTM, Input
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical
from pointerLayer import pointerLayer, scheduler
import pickle
import tsp_data as tsp
import numpy as np
from parameters import BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, NB_EPOCH, TEST_BATCH
# ============================================================================ #
print "# ============================ START ============================ #"
# --------------------- Variables ------------------------------ #
ppath = os.getcwd() + "/"  # Project Path Location
# -------------------------------------------------------------- #

def main():

    # ------------ DATA ---------------- #
    t = tsp.DataGenerator()
    X, Y = t.next_batch(BATCH_SIZE, SEQ_LENGTH)
    X_test, Y_test = t.next_batch(TEST_BATCH, SEQ_LENGTH)
    YY = []
    for y in Y:
        YY.append(to_categorical(y))
    YY = np.asarray(YY)

    # ----------- MODEL ------------- #
    main_input = Input(shape = (SEQ_LENGTH, 2), name = 'main_input')
    encoder = LSTM(output_dim = HIDDEN_SIZE, return_sequences = True, name = "encoder")(main_input)
    decoder = pointerLayer(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, name = "decoder")(encoder)
    model = Model(inputs = main_input, outputs = decoder)
    model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # ---------- TRAIN ------------ #
    model.fit(X, YY, nb_epoch = NB_EPOCH, batch_size = 64,callbacks = [LearningRateScheduler(scheduler),])

    # --------- PREDICT ---------- #
    predictions = model.predict(X_test)
    pred_index = np.array([np.argmax(predictions[i],0) for i in xrange(len(predictions))])
    print pred_index
    print Y_test


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"
    