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

from keras.models import model_from_json
import sort_data as sort
from parameters import TEST_BATCH, SEQ_LENGTH
from pointerLayer import predict

# ============================================================================ #
print "# ============================ START ============================ #"
# --------------------- Variables ------------------------------ #

ppath = os.getcwd() + "/"  # Project Path Location


# -------------------------------------------------------------- #

def main():

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    exit()

    loaded_model.load_weights("./model/weights3.hdf5")


    predict(X_test, Y_test, loaded_model)

# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"
    