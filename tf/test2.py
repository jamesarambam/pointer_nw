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
from dataset import DataGenerator
import numpy as np
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np
from keras.utils.test_utils import keras_test



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

ax.clearDir("./logs")
input_length = 5
input_dim = 3
output_length = 3
output_dim = 4
samples = 100
hidden_dim = 24

# -------------------------------------------------------------- #

@keras_test
def test_SimpleSeq2Seq():
    x = np.random.random((samples, input_length, input_dim))
    y = np.random.random((samples, output_length, output_dim))

    models = []
    models += [SimpleSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim))]
    models += [SimpleSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]



    for model in models:

        print model.train_summary()
        exit()

        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, nb_epoch=1)


def main():


    # --------- Create Dataset ---------- #
    test_SimpleSeq2Seq()









# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"