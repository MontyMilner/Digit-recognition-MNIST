#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 10:05:01 2022

@author: monty
"""
import sys
sys.path.append("/Users/monty/Desktop/ML projects/")
import numpy as np
from MLClass import Gradient_Descent
from mnist import MNIST

def one_hot(array):
    Y=np.zeros((array.size,array.max()+1))
    Y[np.arange(array.size),array]=1
    return Y.T

mndata = MNIST('samples')
images, labels = mndata.load_training()
samples=10005
scale_factor=255
X=(np.array(images[:samples]))/scale_factor
Y=one_hot(np.array(labels[:samples]))
G2=Gradient_Descent(X, Y, [784,100,10],["ReLU","softmax"],500,0.1)
G2.gdescend(True)
