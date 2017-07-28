"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 28 Jul 2017
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
        sys.path.append("/home/arambamjs.2016/projects")
if o == "Darwin":
    sys.path.append("/Users/james/Codes/python")

import auxlib.auxLib as ax

# ================================ priImports ================================ #

from pointerLayer import predict
from keras.models import load_model
import sort_data as sort
from parameters import TEST_BATCH, SEQ_LENGTH
# from pointerLayer import predict
from mainSort import conStruct_Model
import numpy as np

# ============================================================================ #
print "# ============================ START ============================ #"
# --------------------- Variables ------------------------------ #

ppath = os.getcwd() + "/"  # Project Path Location


# -------------------------------------------------------------- #


def predictScore(X_test, Y_test, model):

    pCount = 0
    predictions = model.predict(X_test)
    pred_index = np.array([np.argmax(predictions[i],0) for i in xrange(len(predictions))])
    for i in range(TEST_BATCH):
        x = map(lambda t : t[0], [j for j in X_test[i]])
        y = map(lambda t : t[0], [j for j in Y_test[i]])
        tmp_x = {k: v for v, k in enumerate(x)}
        y_test =  map(lambda t : tmp_x[t], y)
        pred = pred_index[i].tolist()
        if y_test == pred:
            pCount += 1
    return pCount, TEST_BATCH - pCount, (float(pCount) * 100 )/ TEST_BATCH



def predict(X_test, Y_test, model):

    print "\n\n\n"
    print "--------------------------- Prediction --------------------------- "
    print "[X_test]", "[Y_test]", "[Y_Prediction]"
    predictions = model.predict(X_test)
    pred_index = np.array([np.argmax(predictions[i],0) for i in xrange(len(predictions))])
    for i in range(TEST_BATCH):
        x = map(lambda t : t[0], [j for j in X_test[i]])
        y = map(lambda t : t[0], [j for j in Y_test[i]])
        tmp_x = {k: v for v, k in enumerate(x)}
        y_test =  map(lambda t : tmp_x[t], y)
        x_test = map(lambda t : round(t, 4), x)
        pred = pred_index[i].tolist()
        print x_test, y_test, pred, y_test == pred


def main():

    model = conStruct_Model()
    model.load_weights("./model/weights"+str(SEQ_LENGTH)+".hdf5")
    t = sort.DataGenerator()
    print "-------------------------------"
    for i in range(5):
        X_test, Y_test, T_test = t.next_batch(TEST_BATCH, SEQ_LENGTH)
        corr, inco, acc = predictScore(X_test, Y_test, model)
        print "Total :", TEST_BATCH, ", Correct :", corr, ", Incorrect :", inco, ", Accuracy :",  acc, "%"




# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"
    