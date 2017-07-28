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
from keras import callbacks
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical
from pointerLayer import pointerLayer, scheduler, predict
import pickle
import sort_data as sort
import pickle as pkl
import numpy as np
from parameters import DATA_SIZE,BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE, NB_EPOCH, TEST_BATCH
# ============================================================================ #
print "# ============================ START ============================ #"
# --------------------- Variables ------------------------------ #
ppath = os.getcwd() + "/"  # Project Path Location
# -------------------------------------------------------------- #

def conStruct_Model():
    # ----------- MODEL ------------- #
    main_input = Input(shape = (SEQ_LENGTH, 1), name = 'main_input')
    encoder = LSTM(output_dim = HIDDEN_SIZE, return_sequences = True, name = "encoder")(main_input)
    decoder = pointerLayer(HIDDEN_SIZE, output_dim = HIDDEN_SIZE, name = "decoder")(encoder)
    model = Model(inputs = main_input, outputs = decoder)
    return model

def main():

    ax.deleteDir("./tboard")

    # ------------ DATA ---------------- #
    t = sort.DataGenerator()
    X, Y, T = t.next_batch(DATA_SIZE, SEQ_LENGTH)
    X_test, Y_test, T_test = t.next_batch(TEST_BATCH, SEQ_LENGTH)

    # --------------- MODEL ------------------- #

    model = conStruct_Model()
    model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    tbCallBack = callbacks.TensorBoard(log_dir='./tboard/1', histogram_freq=0, write_graph=False, write_images=False)

    # ---------- TRAIN ------------ #
    model.fit(X, T, nb_epoch = NB_EPOCH, batch_size = BATCH_SIZE, callbacks = [tbCallBack, LearningRateScheduler(scheduler),])

    # --------- INFERENCE ---------- #
    predict(X_test, Y_test, model)

    # -------- Save Model -------- #
    model.save_weights("./model/weights"+str(SEQ_LENGTH)+".hdf5")
# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"
    
