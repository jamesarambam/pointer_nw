"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 11 Jun 2017
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
import tensorflow as tf


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

ppath = os.getcwd() + "/"  # Project Path Location


# -------------------------------------------------------------- #

def main():


    with tf.name_scope('cell') as scope:
        cell = tf.contrib.rnn.GRUCell(2)


    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    writer = tf.summary.FileWriter("./logs/1")
    writer.add_graph(sess.graph)




# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"