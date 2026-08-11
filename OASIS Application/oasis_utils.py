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