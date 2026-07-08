import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random
import types

from time import time

from scipy.special import comb, loggamma, lambertw
from scipy.stats import multinomial, expon

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import optimizers, initializers, regularizers, layers

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

import os, shutil
from pathlib import Path
import json
import subprocess

import thetaflow as thf

import pwexp

def initialize_alpha_s(t, delta, n_cuts = 6):
    alpha0 = np.ones(n_cuts + 1)
    qs = np.linspace(0, 1, n_cuts+2)[1:-1]
    s = np.quantile(t[delta == 1], qs)
    s = np.concatenate([[0],s])
    alpha0 = tf.cast(alpha0, tf.float32)
    s = tf.cast(s, tf.float32)
    return alpha0, s

def build_mpscr_model(y, delta, input_dim, model_spec, seed = 10, n_cuts = 5):
    '''
        Structure the Two-parameter Modified Power Series distribution in the architecture required for the thetaflow package.
        A model_spec object is provided, which is aimed at completely specifying the modified power series model by providing its associated functions
        such as, a_m(q), phi(theta, q), C(theta, q), as well as its inherent probability bounds:
        While more flexible models have simply p in (0,1), more specific models such as the Borel distribution admits p in (1/e, 1) only, being physically unable to produce cure probabilities interior of p_min.
        
        For the baseline distribution, we assume here to be a simple Piecewise-Exponential, following the work of Xie & Yu (2021). We keep 5 cuts by default. However, that number can be semalessly changed.
    '''

    a0 = model_spec.a0
    phi = model_spec.phi
    phi_inv = model_spec.phi_inv
    C = model_spec.C
    C_inv = model_spec.C_inv
    p_min = model_spec.p_min
    p_max = model_spec.p_max
    hasq = model_spec.hasq
    
    _, s = initialize_alpha_s(y, delta, n_cuts = n_cuts)

    def softplus_inv(y):
        return tf.math.log(tf.math.exp(y) - 1)

    def bounded_sigmoid(u):
        return tf.math.sigmoid(u) * (p_max - p_min) + p_min
    
    def logit(u):
        return -( tf.math.log(1-u) - tf.math.log(u) )

    def bounded_logit(u):
        return logit( (u - p_min) / (p_max - p_min) )

    if(hasq):
        parameters = {
            "alpha": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "independent", "shape": n_cuts+1, "init": 1.0, "warmup_time": 0},
            "q": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "independent", "shape": 1, "init": 1.0, "warmup_time": 0},
            "p": {"link": bounded_sigmoid, "link_inv": bounded_logit, "par_type": "nn", "shape": 1, "init": 0.5, "warmup_time": 0},
        }
    else:
        parameters = {
            "alpha": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "independent", "shape": n_cuts+1, "init": 1.0, "warmup_time": 0},
            "p": {"link": bounded_sigmoid, "link_inv": bounded_logit, "par_type": "nn", "shape": 1, "init": 0.5, "warmup_time": 0},
        }

    def A(u, q):
        '''
            Evaluates the intermediate function A(u; q) based on the provided normalized constant, C(theta; q).
            Since C(theta; q) = A(phi(theta; q)), A(u; q) = C(phi_inv(theta; q); q)
        '''
        theta = phi_inv(u, q)
        return C(theta, q)
    
    def loglikelihood_loss(model, nn_output, data):
        X, y, delta = data
        
        # alpha is the vector of parameters from the base distribution (piecewise exponential in this case)
        alpha = model.get_variable("alpha")

        if(model.hasq):
            # q is a constant parameter (e.g. dispersion of a Negative Binomial)
            q = model.get_variable("q")
        else:
            q = 0.0
        # Theta represents the lead parameter of the model (e.g. scale of a Negative Binomial)
        p = model.get_variable("p", nn_output)

        eps = tf.constant(1.0e-5, dtype = tf.float32)
        y = tf.clip_by_value(y, eps, np.inf)
        p = tf.clip_by_value(p, eps, 1.0-eps)
        
        theta = C_inv( a0(q) / p, q )

        # Base survival function (piecewise exponential in this case)
        S0 = pwexp.cdf(y, alpha, s, lower_tail = False)
        log_f0 = tf.math.log( pwexp.pdf(y, alpha, s) )
        
        C_theta = C( theta, q )

        u = S0 * phi(theta, q)
        with tf.GradientTape() as tape:
            tape.watch(u)
            A_u = A( u, q )
        
        log_S_pop = tf.math.log( A_u ) - tf.math.log( C_theta )
        Aprime_u = tape.gradient(A_u, u)
        log_f_pop = tf.math.log( Aprime_u ) - tf.math.log( C_theta ) + log_f0 + tf.math.log( phi(theta, q) )
        
        loglik_terms = delta * log_f_pop + (1-delta) * log_S_pop
        # loglik_terms = tf.constant([0.0], dtype = tf.float32)
        neg_loglik = -tf.reduce_sum(loglik_terms)
        
        return neg_loglik

    def neural_network(model, seed = None):
        initializer = initializers.GlorotNormal(seed = seed)
        model.dense1 = layers.Dense(
            units = 16,
            activation = "softplus",
            kernel_initializer = initializer,
            use_bias = True,
            dtype = tf.float32
        )
        model.dense2 = layers.Dense(
            units = 8,
            activation = "softplus",
            kernel_initializer = initializer,
            use_bias = True,
            dtype = tf.float32
        )
        model.output_layer = layers.Dense(
            units = 1,
            activation = None,
            use_bias = True,
            kernel_initializer = initializer,
            dtype = tf.float32
        )
    
    def neural_network_call(model, x_input, training = False):
        x = model.dense1(x_input)
        x = model.dense2(x)
        x = model.output_layer(x)
        return x
    
    def neural_network_call_nolast(model, x_input):
        x = model.dense1(x_input)
        x = model.dense2(x)
        x = model.dense3(x)
        return x

    model = thf.ModelNN(parameters, loglikelihood_loss,
                        neural_network, neural_network_call,
                        neural_network_call_nolast, input_dim = input_dim, seed = seed)

    model.hasq = hasq
    model.s = s
    model.a0 = a0
    model.phi = phi
    model.phi_inv = phi_inv
    model.C = C
    model.C_inv = C_inv
    model.p_min = p_min
    model.p_max = p_max
    model.A = A

    def get_survival_cure(self, y_train, X_train, y_test, X_test, ngrid = 100):    
        pred_train = self.predict(X_train)
        pred_test = self.predict(X_test)
        
        alpha = self.predict("alpha")
        if(self.hasq):
            q = self.predict("q")
        else:
            q = 0.0
        s = self.s
    
        ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]
        
        eps = tf.constant(1.0e-5, dtype = tf.float32)
        p_train = pred_train["p"].numpy().flatten()
        p_test = pred_test["p"].numpy().flatten()
        p_train = tf.clip_by_value(p_train, eps, 1.0-eps)
        p_test = tf.clip_by_value(p_test, eps, 1.0-eps)
    
        theta_train = self.C_inv( a0(q) / p_train, q )
        theta_test = self.C_inv( a0(q) / p_test, q )
    
        S0_ts = pwexp.cdf(ts_grid, alpha, s, lower_tail = False)
        S0_train = pwexp.cdf(y_train, alpha, s, lower_tail = False)
        S0_test = pwexp.cdf(y_test, alpha, s, lower_tail = False)
        
        u_ts_train = S0_ts * self.phi(theta_train, q)
        u_ts_test = S0_ts * self.phi(theta_test, q)
        u_train = S0_train * self.phi(theta_train, q)
        u_test = S0_test * self.phi(theta_test, q)
        
        A_u_ts_train = self.A( u_ts_train, q )
        A_u_ts_test = self.A( u_ts_test, q )
        A_u_train = self.A( u_train, q )
        A_u_test = self.A( u_test, q )
    
        C_theta_train = C( theta_train, q )
        C_theta_test = C( theta_test, q )
        
        S_ts_train = A_u_ts_train / C_theta_train
        S_ts_test = A_u_ts_test / C_theta_test
        S_train = A_u_train / C_theta_train
        S_test = A_u_test / C_theta_test
        
        H_train = -np.log( S_train )
        H_test = -np.log( S_test )
    
        return {
            "ts_grid": ts_grid,
            "S_ts_train": S_ts_train,
            "S_ts_test": S_ts_test,
            "S_train": S_train,
            "S_test": S_test,
            "H_train": H_train,
            "H_test": H_test,
            "theta_train": theta_train,
            "theta_test": theta_test,
            "p_train": p_train,
            "p_test": p_test,
            "alpha": alpha,
            "s": self.s
        }

    model.get_survival_cure = types.MethodType( get_survival_cure, model )
    
    return model

class MPSPoisson:

    def __init__(self):
        self.p_min = 0.0
        self.p_max = 1.0
        self.sup = np.arange(501)
        self.hasq = False

        def a(m, q):
            return tf.math.exp( self.log_a(m, q) )
    
        def a0(q):
            return 1.0
        
        def log_a(m, q):
            return -tf.math.lgamma(m+1)
        
        def phi(theta, q):
            return tf.identity(theta)
        
        def log_phi(theta, q):
            return tf.math.log(theta)
                               
        def phi_inv(u, q):
            return tf.identity(u)
        
        def C(theta, q):
            return tf.math.exp(theta)
        
        def C_inv(u, q):
            return tf.math.log(u)
        
        def A(u, q):
            theta = phi_inv(u, q)
            return C(theta, q)

        self.a = a
        self.a0 = a0
        self.log_a = log_a
        self.phi = phi
        self.log_phi = log_phi
        self.phi_inv = phi_inv
        self.C = C
        self.C_inv = C_inv
        self.A = A


class MPSBinomial:

    def __init__(self):
        a0 = 1.0
        p_min = 0.0
        p_max = 1.0
        sup = np.arange(501)

    def a(m, q):
        return tf.math.exp( self.log_a(m, q) )

    def a0(q):
        return 1.0
    
    def log_a(m, q):
        return -tf.math.lgamma(m+1)
    
    def phi(theta, q):
        return tf.identity(theta)
    
    def log_phi(theta, q):
        return tf.math.log(theta)
                           
    def phi_inv(u, q):
        return tf.identity(u)
    
    def C(theta, q):
        return tf.math.exp(theta)
    
    def C_inv(u, q):
        return tf.math.log(u)
    
    def A(u, q):
        theta = phi_inv(u, q)
        return C(theta, q)

    self.a = a
    self.a0 = a0
    self.log_a = log_a
    self.phi = phi
    self.log_phi = log_phi
    self.phi_inv = phi_inv
    self.C = C
    self.C_inv = C_inv
    self.A = A

















        