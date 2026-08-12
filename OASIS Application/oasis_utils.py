import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import random

from time import time

from scipy.special import comb, loggamma, lambertw
from scipy.stats import multinomial, expon

import lifelines
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import optimizers, initializers, regularizers, layers

import os, shutil, sys
from pathlib import Path
import json
import subprocess

from sklearn.model_selection import train_test_split

def load_oasis_data(long_term_limit = None, train_size = 0.75, random_state = 15, stratify = True, verbose = True):
    """
        Load the preprocessed OASIS data and automatically split it into a train and a test sets.
        Consider 
    """

    df = pd.read_csv("OASIS Data/patients_full_data.csv")
    x = np.load("OASIS Data/patients_images.npy")
    x = np.expand_dims(x, axis = -1)
    
    # If provided, apply administrative Type I censor to all patient's times
    if(long_term_limit is not None):
        long_term_patients = (df["time_days"] / 365) > long_term_limit
        long_term_events = long_term_patients & (df["delta"] == 1)
        if(verbose):
            print("Applying administrative Type I censorship in {} patients, of which {} had observed events.".format(long_term_patients.sum(), long_term_events.sum()))
        df.loc[long_term_patients, "time_days"] = long_term_limit * 365
        df.loc[long_term_patients, "delta"] = 0
    
    indices = np.arange(df.shape[0])
    if(stratify):
        idx_train, idx_test = train_test_split(indices, train_size = train_size, random_state = random_state, stratify = df["delta"])
    else:
        idx_train, idx_test = train_test_split(indices, train_size = train_size, random_state = random_state)
    
    print("Train dimension: {}".format(idx_train.shape))
    print("Test dimension: {}".format(idx_test.shape))
    
    y_train = df.loc[idx_train, "time_days"] / 365
    y_test = df.loc[idx_test, "time_days"] / 365
    delta_train = df.loc[idx_train, "delta"]
    delta_test = df.loc[idx_test, "delta"]
    
    x_train = x[idx_train]
    x_test = x[idx_test]

    return y_train, delta_train, x_train, y_test, delta_test, x_test


def build_neural_network_structure(n_outputs):
    """
        Builds our rigid defined neural network structure where the number of output parameters is specified manually.
        That allows us to use this same function to call the different model architectures we propose here.
    """
    def neural_network(model, seed = None):
        initializer = initializers.GlorotNormal(seed = seed)
        model.convolution1 = keras.layers.Conv2D(filters = 4, kernel_size = [7,7], padding = "same", activation = tf.nn.leaky_relu,
                                                 kernel_initializer = initializer, dtype = tf.float32)
        model.pooling1 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        model.convolution2 = keras.layers.Conv2D(filters = 8, kernel_size = [5,5], padding = "same", activation = tf.nn.leaky_relu,
                                                 kernel_initializer = initializer, dtype = tf.float32)
        model.pooling2 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        model.convolution3 = keras.layers.Conv2D(filters = 16, kernel_size = [5,5], padding = "same", activation = tf.nn.leaky_relu,
                                                 kernel_initializer = initializer, dtype = tf.float32)
        model.pooling3 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        model.convolution4 = keras.layers.Conv2D(filters = 32, kernel_size = [3,3], padding = "same", activation = tf.nn.leaky_relu, kernel_initializer = initializer, dtype = tf.float32)
        model.pooling4 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        model.convolution5 = keras.layers.Conv2D(filters = 64, kernel_size = [3,3], padding = "same", activation = tf.nn.leaky_relu, kernel_initializer = initializer, dtype = tf.float32)
        model.pooling5 = keras.layers.MaxPool2D(pool_size = [2,2], strides = 2)
        model.flatten = keras.layers.Reshape(target_shape=(-1,))
        model.dense1 = keras.layers.Dense(units = 128, activation = tf.nn.leaky_relu, kernel_initializer = initializer, dtype = tf.float32)
        model.dense2 = keras.layers.Dense(units = 4, activation = tf.nn.leaky_relu, kernel_initializer = initializer, dtype = tf.float32)
        model.dense3 = keras.layers.Dense(units = n_outputs, kernel_initializer = initializer, dtype = tf.float32, activation = None, use_bias = False)

    def neural_network_call(model, x_input, training = False):
        x = model.convolution1(x_input)
        x = model.pooling1(x)
        x = model.convolution2(x)
        x = model.pooling2(x)
        x = model.convolution3(x)
        x = model.pooling3(x)
        x = model.convolution4(x)
        x = model.pooling4(x)
        x = model.convolution5(x)
        x = model.pooling5(x)
        x = model.flatten(x)
        x = model.dense1(x)
        x = model.dense2(x)
        x = model.dense3(x)
        return x

    def neural_network_call_nolast(model, x_input):
        x = model.convolution1(x_input)
        x = model.pooling1(x)
        x = model.convolution2(x)
        x = model.pooling2(x)
        x = model.convolution3(x)
        x = model.pooling3(x)
        x = model.convolution4(x)
        x = model.pooling4(x)
        x = model.convolution5(x)
        x = model.pooling5(x)
        x = model.flatten(x)
        x = model.dense1(x)
        x = model.dense2(x)
        return x
        
    return neural_network, neural_network_call, neural_network_call_nolast