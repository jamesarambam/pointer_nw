"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 27 Jul 2017
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

o = platform.system()
if o == "Linux":
    d = platform.dist()
    if d[0] == "debian":
        sys.path.append("/home/james/Codes/python")
    if d[0] == "centos":
        sys.path.append("/home/arambamjs.2016/project")
if o == "Darwin":
    sys.path.append("/Users/james/Codes/python")
import auxlib.auxLib as ax

# ================================ secImports ================================ #

from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================ #

print "# ============================ START ============================ #"

# --------------------- Variables ------------------------------ #

ppath = os.getcwd() + "/"  # Project Path Location


# -------------------------------------------------------------- #

def main():


    acc = genfromtxt('results/acc.csv', delimiter=',')
    loss = genfromtxt('results/loss.csv', delimiter=',')

    dataSize = len(acc)

    dataSize = dataSize-0



    X = []
    Y_acc = []
    Y_loss = []
    for i in range(1, dataSize):
        X.append(acc[i][1])
        Y_acc.append(acc[i][2])
        Y_loss.append(loss[i][2])

    plt.subplot(211)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace= 0.2, hspace = 0.9)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(Y_acc)
    plt.legend()
    plt.subplot(212)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.09, wspace= 0.2, hspace = 0.9)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(Y_loss)
    plt.legend()
    plt.show()


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"
