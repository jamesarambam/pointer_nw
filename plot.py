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


    X = []
    Y_acc = []
    Y_loss = []
    for i in range(1, 1001):
        X.append(acc[i][1])
        Y_acc.append(acc[i][2])
        Y_loss.append(loss[i][2])

    plt.subplot(211)
    plt.plot(Y_acc, label="Accuracy")
    plt.legend()
    plt.subplot(212)
    plt.plot(Y_loss, label="Loss")
    plt.legend()
    plt.show()


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"