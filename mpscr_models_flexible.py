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

import pwexp_flexible as pwexp

def initialize_alpha_s(t, delta, n_cuts = 6):
    alpha0 = np.ones(n_cuts + 1)
    qs = np.linspace(0, 1, n_cuts+2)[1:-1]
    s = np.quantile(t[delta == 1], qs)
    s = np.concatenate([[0],s])
    alpha0 = tf.cast(alpha0, tf.float32)
    s = tf.cast(s, tf.float32)
    return alpha0, s

def build_simple_mpscr_model(y, delta, model_spec, base_spec, seed = 10):
    '''
        Structure the Two-parameter Modified Power Series distribution in the architecture of thetaflow, considering the
        total absence of predicting variables. While the model below admits input data and a neural network structure as inputs,
        the model build here simply considers all parameters as populational constants. We aim to use this model to show that
        the image model in OASIS-3 is indeed capturing relevant patterns for prediction, capturing more than simply random noise.
    '''
    # Basic functions that define a specific distribution from the Two-Parameter Modified Power Series family 
    a0 = model_spec.a0
    phi = model_spec.phi
    phi_inv = model_spec.phi_inv
    C = model_spec.C
    C_inv = model_spec.C_inv
    # Functions of q to define the parameter space of the cure probability, which models theta directly
    # In special cases of the MPS family, such as the Restricted Generalized Poisson (RGP) model,
    # we have that, for a given value of q, theta varies in the range (0, |q|^{-1}),
    # therefore, we have a varying parameter space, that must be updated with each new value of q
    p_min = model_spec.p_min
    p_max = model_spec.p_max
    # If hasq is True, the model treats q as an unknown parameter to be estimated
    hasq = model_spec.hasq
    # If hasq is False, then q is treated as a known constant or is completely ignored
    # In the first case, fixed_q represents that known value
    fixed_q = model_spec.fixed_q
    # If q is a trainable parameter, must specify its corresponding link function
    link_q = None
    link_inv_q = None
    A = model_spec.A
    if(hasq):
        link_q = model_spec.link_q
        link_inv_q = model_spec.link_inv_q

    def softplus_inv(u):
        return tf.math.log(tf.math.exp(u) - 1)
    
    if(hasq):
        parameters = {
            "alpha": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "independent", "shape": base_spec.n_parameters, "init": 1.0, "warmup_time": 0},
            "q": {"link": link_q, "link_inv": link_inv_q, "par_type": "independent", "shape": 1, "init": 1.0, "warmup_time": 0},
            "raw_p": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": 1, "init": 0.0, "warmup_time": 0},
        }
    else:
        parameters = {
            "alpha": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "independent", "shape": base_spec.n_parameters, "init": 1.0, "warmup_time": 0},
            "raw_p": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": 1, "init": 0.0, "warmup_time": 0},
        }
    
    def loglikelihood_loss(model, nn_output, data):
        X, y, delta = data
        
        # alpha is the vector of parameters from the base distribution (piecewise exponential in this case)
        alpha = model.get_variable("alpha")[None,:]
        
        if(model.hasq):
            # q is a constant parameter (e.g. dispersion of a Negative Binomial)
            q = model.get_variable("q")
        else:
            q = fixed_q
            
        # Theta represents the lead parameter of the model (e.g. scale of a Negative Binomial)
        raw_p = model.get_variable("raw_p")
        p = tf.math.sigmoid(raw_p) * (p_max(q) - p_min(q)) + p_min(q)
        
        eps = tf.constant(1.0e-4, dtype = tf.float32)
        y = tf.clip_by_value(y, eps, np.inf)
        p = tf.clip_by_value(p, eps, 1.0-eps)
        
        theta = C_inv( a0(q) / p, q )
        
        # Base survival function (piecewise exponential in this case)
        S0 = tf.reshape( base_spec.survival(y, alpha), [-1,1] )
        log_h0 = tf.reshape( base_spec.log_h(y, alpha), [-1,1] )
        log_S0 = tf.math.log( S0 )
        log_f0 = log_h0 + log_S0

        C_theta = C( theta, q )
        
        u = S0 * phi(theta, q)
        with tf.GradientTape() as tape:
            tape.watch(u)
            A_u = A( u, q )

        log_S_pop = tf.math.log( A_u ) - tf.math.log( C_theta )
        Aprime_u = tape.gradient(A_u, u)
        log_f_pop = tf.math.log( Aprime_u ) - tf.math.log( C_theta ) + log_f0 + tf.math.log( phi(theta, q) )

        loglik_terms = delta * log_f_pop + (1-delta) * log_S_pop
        
        neg_loglik = -tf.reduce_sum(loglik_terms)
        
        return neg_loglik

    model = thf.ModelNN(parameters, loglikelihood_loss,
                        None, None, None,
                        input_dim = (1,), seed = seed)
    model.hasq = hasq
    model.a0 = a0
    model.phi = phi
    model.phi_inv = phi_inv
    model.C = C
    model.C_inv = C_inv
    model.p_min = p_min
    model.p_max = p_max
    model.A = A
    model.link_q = link_q
    model.link_inv_q = link_inv_q
    
    def get_survival_cure(self, y_train, delta_train, y_test, delta_test, ngrid = 100):    
        alpha = self.predict("alpha")[None,:]
        
        if(self.hasq):
            q = self.predict("q")
        else:
            q = fixed_q
    
        ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)
        
        eps = tf.constant(1.0e-5, dtype = tf.float32)
        raw_p = self.predict("raw_p")

        p = tf.math.sigmoid(raw_p) * (self.p_max(q) - self.p_min(q)) + self.p_min(q)
        p = tf.clip_by_value(p, eps, 1.0-eps).numpy()
    
        theta = self.C_inv( self.a0(q) / p, q )
    
        S0_ts = tf.cast( base_spec.survival(ts_grid, alpha), tf.float32 )
        S0_train = tf.cast( base_spec.survival(y_train, alpha), tf.float32 )
        S0_test = tf.cast( base_spec.survival(y_test, alpha), tf.float32 )
        
        u_ts = S0_ts * self.phi(theta, q)
        u_train = S0_train * self.phi(theta, q)
        u_test = S0_test * self.phi(theta, q)
        
        A_u_ts = self.A( u_ts, q )
        A_u_train = self.A( u_train, q )
        A_u_test = self.A( u_test, q )
    
        C_theta = self.C( theta, q )
        
        S_ts_train = A_u_ts / C_theta
        S_ts_test = A_u_ts / C_theta
        S_train = A_u_train / C_theta
        S_test = A_u_test / C_theta
        
        H_train = -np.log( S_train )
        H_test = -np.log( S_test )

        if(hasattr(y_train, "to_numpy")):
            y_train = y_train.to_numpy()
        if(hasattr(delta_train, "to_numpy")):
            delta_train = delta_train.to_numpy()
        if(hasattr(y_test, "to_numpy")):
            y_test = y_test.to_numpy()
        if(hasattr(delta_test, "to_numpy")):
            delta_test = delta_test.to_numpy()

        results_dict = {
            "ts_grid": ts_grid,
            "S_ts_train": S_ts_train,
            "S_ts_test": S_ts_test,
            "y_train": y_train,
            "y_test": y_test,
            "delta_train": delta_train,
            "delta_test": delta_test,
            "S_train": S_train,
            "S_test": S_test,
            "H_train": H_train,
            "H_test": H_test,
            "theta_train": theta,
            "theta_test": theta,
            "p_train": p,
            "p_test": p,
            "alpha": alpha
        }
        # Runs through all elements in the results dictionary and convert them to numpy, if possible
        for key in results_dict:
            if(hasattr(results_dict[key], "to_numpy")):
                results_dict[key] = results_dict[key].to_numpy()
            if(hasattr(results_dict[key], "numpy")):
                results_dict[key] = results_dict[key].numpy()
        
        return results_dict
    
    model.get_survival_cure = types.MethodType( get_survival_cure, model )
    
    return model


def build_medium_mpscr_model(y, delta, input_dim, model_spec, base_spec,
                             neural_network = None, neural_network_call = None, neural_network_call_nolast = None,
                             seed = 10):
    """
        Structure the Two-parameter Modified Power Series distribution promotion time cure model using the thetaflow package.
        
        Here, we consider the base distribution's parameter vector alpha is a global constant and only the cure probability of each
        patient is modeled according to an arbitrary neural network.
    """
    
    # Basic functions that define a specific distribution from the Two-Parameter Modified Power Series family 
    a0 = model_spec.a0
    phi = model_spec.phi
    phi_inv = model_spec.phi_inv
    C = model_spec.C
    C_inv = model_spec.C_inv
    # Functions of q to define the parameter space of the cure probability, which models theta directly
    # In special cases of the MPS family, such as the Restricted Generalized Poisson (RGP) model,
    # we have that, for a given value of q, theta varies in the range (0, |q|^{-1}),
    # therefore, we have a varying parameter space, that must be updated with each new value of q
    p_min = model_spec.p_min
    p_max = model_spec.p_max
    # If hasq is True, the model treats q as an unknown parameter to be estimated
    hasq = model_spec.hasq
    # If hasq is False, then q is treated as a known constant or is completely ignored
    # In the first case, fixed_q represents that known value
    fixed_q = model_spec.fixed_q
    # If q is a trainable parameter, must specify its corresponding link function
    link_q = None
    link_inv_q = None
    A = model_spec.A
    if(hasq):
        link_q = model_spec.link_q
        link_inv_q = model_spec.link_inv_q
    
    def softplus_inv(u):
        return tf.math.log(tf.math.exp(u) - 1)
    
    if(hasq):
        parameters = {
            "alpha": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "independent", "shape": base_spec.n_parameters, "init": 1.0, "warmup_time": 0},
            "q": {"link": link_q, "link_inv": link_inv_q, "par_type": "independent", "shape": 1, "init": 1.0, "warmup_time": 0},
            "raw_p": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": 0.0, "warmup_time": 0},
        }
    else:
        parameters = {
            "alpha": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "independent", "shape": base_spec.n_parameters, "init": 1.0, "warmup_time": 0},
            "raw_p": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": 0.0, "warmup_time": 0},
        }
    
    def loglikelihood_loss(model, nn_output, data):
        X, y, delta = data
        
        # alpha is the vector of parameters from the base distribution (piecewise exponential in this case)
        alpha = model.get_variable("alpha")[None,:]

        if(model.hasq):
            # q is a constant parameter (e.g. dispersion of a Negative Binomial)
            q = model.get_variable("q")
        else:
            q = fixed_q
        # Theta represents the lead parameter of the model (e.g. scale of a Negative Binomial)
        raw_p = model.get_variable("raw_p", nn_output)
        p = tf.math.sigmoid(raw_p) * (p_max(q) - p_min(q)) + p_min(q)
        
        eps = tf.constant(1.0e-4, dtype = tf.float32)
        y = tf.clip_by_value(y, eps, np.inf)
        p = tf.clip_by_value(p, eps, 1.0-eps)
        
        theta = C_inv( a0(q) / p, q )
        
        # Base survival function (piecewise exponential in this case)
        S0 = tf.reshape( base_spec.survival(y, alpha), [-1,1] )
        log_h0 = tf.reshape( base_spec.log_h(y, alpha), [-1,1] )        
        log_S0 = tf.math.log( S0 )
        log_f0 = log_h0 + log_S0
        
        C_theta = C( theta, q )

        u = S0 * phi(theta, q)
        with tf.GradientTape() as tape:
            tape.watch(u)
            A_u = A( u, q )
        
        log_S_pop = tf.math.log( A_u ) - tf.math.log( C_theta )
        Aprime_u = tape.gradient(A_u, u)
        log_f_pop = tf.math.log( Aprime_u ) - tf.math.log( C_theta ) + log_f0 + tf.math.log( phi(theta, q) )
        
        loglik_terms = delta * log_f_pop + (1-delta) * log_S_pop
        neg_loglik = -tf.reduce_sum(loglik_terms)
        
        return neg_loglik

    model = thf.ModelNN(parameters, loglikelihood_loss,
                        neural_network, neural_network_call,
                        neural_network_call_nolast, input_dim = input_dim, seed = seed)
    model.hasq = hasq
    model.a0 = a0
    model.phi = phi
    model.phi_inv = phi_inv
    model.C = C
    model.C_inv = C_inv
    model.p_min = p_min
    model.p_max = p_max
    model.A = A
    model.link_q = link_q
    model.link_inv_q = link_inv_q
    
    def get_survival_cure(self, y_train, delta_train, X_train, y_test, delta_test, X_test, ngrid = 100):    
        pred_train = self.predict(X_train)
        pred_test = self.predict(X_test)
        
        alpha = self.predict("alpha")[None,:]
        if(self.hasq):
            q = self.predict("q")
        else:
            q = fixed_q
    
        ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]
        
        eps = tf.constant(1.0e-5, dtype = tf.float32)
        raw_p_train = pred_train["raw_p"].numpy()
        raw_p_test = pred_test["raw_p"].numpy()

        p_train = tf.math.sigmoid(raw_p_train) * (self.p_max(q) - self.p_min(q)) + self.p_min(q)
        p_test = tf.math.sigmoid(raw_p_test) * (self.p_max(q) - self.p_min(q)) + self.p_min(q)

        p_train = tf.clip_by_value(p_train, eps, 1.0-eps)
        p_test = tf.clip_by_value(p_test, eps, 1.0-eps)
    
        theta_train = self.C_inv( self.a0(q) / p_train, q )
        theta_test = self.C_inv( self.a0(q) / p_test, q )
    
        S0_ts = tf.cast( base_spec.survival(ts_grid, alpha), tf.float32 )
        S0_train = tf.cast( base_spec.survival(y_train, alpha), tf.float32 )
        S0_test = tf.cast( base_spec.survival(y_test, alpha), tf.float32 )
        
        u_ts_train = S0_ts * self.phi(theta_train, q)
        u_ts_test = S0_ts * self.phi(theta_test, q)
        u_train = S0_train * self.phi(theta_train, q)
        u_test = S0_test * self.phi(theta_test, q)
        
        A_u_ts_train = self.A( u_ts_train, q )
        A_u_ts_test = self.A( u_ts_test, q )
        A_u_train = self.A( u_train, q )
        A_u_test = self.A( u_test, q )
    
        C_theta_train = self.C( theta_train, q )
        C_theta_test = self.C( theta_test, q )
        
        S_ts_train = A_u_ts_train / C_theta_train
        S_ts_test = A_u_ts_test / C_theta_test
        S_train = A_u_train / C_theta_train
        S_test = A_u_test / C_theta_test
        
        H_train = -np.log( S_train )
        H_test = -np.log( S_test )
    
        results_dict = {
            "ts_grid": ts_grid,
            "S_ts_train": S_ts_train,
            "S_ts_test": S_ts_test,
            "y_train":y_train,
            "y_test": y_test,
            "delta_train": delta_train,
            "delta_test": delta_test,
            "S_train": S_train,
            "S_test": S_test,
            "H_train": H_train,
            "H_test": H_test,
            "theta_train": theta_train,
            "theta_test": theta_test,
            "p_train": p_train,
            "p_test": p_test,
            "alpha": alpha
        }
        # Runs through all elements in the results dictionary and convert them to numpy, if possible
        for key in results_dict:
            if(hasattr(results_dict[key], "to_numpy")):
                results_dict[key] = results_dict[key].to_numpy()
            if(hasattr(results_dict[key], "numpy")):
                results_dict[key] = results_dict[key].numpy()
        
        return results_dict

    model.get_survival_cure = types.MethodType( get_survival_cure, model )
    
    return model


def build_flexible_mpscr_model(y, delta, input_dim, model_spec, base_spec,
                               neural_network = None, neural_network_call = None, neural_network_call_nolast = None,
                               seed = 10):
    '''
        Structure the Two-parameter Modified Power Series distribution in the architecture required for the thetaflow package.
        A model_spec object is provided, which is aimed at completely specifying the modified power series model by providing its associated functions
        such as, a_m(q), phi(theta, q), C(theta, q), as well as its inherent probability bounds:
        While more flexible models have simply p in (0,1), more specific models such as the Borel distribution admits p in (1/e, 1) only, being physically unable to produce cure probabilities interior of p_min.
        
        For the baseline distribution, we assume here to be a simple Piecewise-Exponential, following the work of Xie & Yu (2021). We keep 5 cuts by default. However, that number can be semalessly changed.
    '''

    # Basic functions that define a specific distribution from the Two-Parameter Modified Power Series family 
    a0 = model_spec.a0
    phi = model_spec.phi
    phi_inv = model_spec.phi_inv
    C = model_spec.C
    C_inv = model_spec.C_inv
    # Functions of q to define the parameter space of the cure probability, which models theta directly
    # In special cases of the MPS family, such as the Restricted Generalized Poisson (RGP) model,
    # we have that, for a given value of q, theta varies in the range (0, |q|^{-1}),
    # therefore, we have a varying parameter space, that must be updated with each new value of q
    p_min = model_spec.p_min
    p_max = model_spec.p_max
    # If hasq is True, the model treats q as an unknown parameter to be estimated
    hasq = model_spec.hasq
    # If hasq is False, then q is treated as a known constant or is completely ignored
    # In the first case, fixed_q represents that known value
    fixed_q = model_spec.fixed_q
    # If q is a trainable parameter, must specify its corresponding link function
    link_q = None
    link_inv_q = None
    A = model_spec.A
    if(hasq):
        link_q = model_spec.link_q
        link_inv_q = model_spec.link_inv_q
    
    def softplus_inv(u):
        return tf.math.log(tf.math.exp(u) - 1)
    
    if(hasq):
        parameters = {
            "alpha": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": base_spec.n_parameters, "init": 1.0, "warmup_time": 0},
            "q": {"link": link_q, "link_inv": link_inv_q, "par_type": "independent", "shape": 1, "init": 1.0, "warmup_time": 0},
            "raw_p": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": 0.0, "warmup_time": 0},
        }
    else:
        parameters = {
            "alpha": {"link": tf.math.softplus, "link_inv": softplus_inv, "par_type": "nn", "shape": base_spec.n_parameters, "init": 1.0, "warmup_time": 0},
            "raw_p": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": 0.0, "warmup_time": 0},
        }
    
    def loglikelihood_loss(model, nn_output, data):
        X, y, delta = data
        
        # alpha is the vector of parameters from the base distribution (piecewise exponential in this case)
        alpha = model.get_variable("alpha", nn_output)
        
        if(model.hasq):
            # q is a constant parameter (e.g. dispersion of a Negative Binomial)
            q = model.get_variable("q")
        else:
            q = fixed_q
        # Theta represents the lead parameter of the model (e.g. scale of a Negative Binomial)
        raw_p = model.get_variable("raw_p", nn_output)
        p = tf.math.sigmoid(raw_p) * (p_max(q) - p_min(q)) + p_min(q)
        
        eps = tf.constant(1.0e-4, dtype = tf.float32)
        y = tf.clip_by_value(y, eps, np.inf)
        p = tf.clip_by_value(p, eps, 1.0-eps)
        
        theta = C_inv( a0(q) / p, q )

        # Base survival function (piecewise exponential in this case)
        S0 = base_spec.survival(y, alpha)
        log_h0 = base_spec.log_h(y, alpha)
        log_S0 = tf.math.log( S0 )
        log_f0 = log_h0 + log_S0
        
        C_theta = C( theta, q )
            
        u = S0 * phi(theta, q)
        with tf.GradientTape() as tape:
            tape.watch(u)
            A_u = A( u, q )
        
        log_S_pop = tf.math.log( A_u ) - tf.math.log( C_theta )
        Aprime_u = tape.gradient(A_u, u)
        log_f_pop = tf.math.log( Aprime_u ) - tf.math.log( C_theta ) + log_f0 + tf.math.log( phi(theta, q) )
        
        loglik_terms = delta * log_f_pop + (1-delta) * log_S_pop
        neg_loglik = -tf.reduce_sum(loglik_terms)
        
        return neg_loglik

    model = thf.ModelNN(parameters, loglikelihood_loss,
                        neural_network, neural_network_call,
                        neural_network_call_nolast, input_dim = input_dim, seed = seed)
    model.hasq = hasq
    model.a0 = a0
    model.phi = phi
    model.phi_inv = phi_inv
    model.C = C
    model.C_inv = C_inv
    model.p_min = p_min
    model.p_max = p_max
    model.A = A
    model.link_q = link_q
    model.link_inv_q = link_inv_q
    
    def get_survival_cure(self, y_train, delta_train, X_train, y_test, delta_test, X_test, ngrid = 100):    
        pred_train = self.predict(X_train)
        pred_test = self.predict(X_test)
        
        # alpha = self.predict("alpha")
        alpha_train = pred_train["alpha"]
        alpha_test = pred_test["alpha"]
        
        if(self.hasq):
            q = self.predict("q")
        else:
            q = fixed_q
    
        ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)
        
        eps = tf.constant(1.0e-5, dtype = tf.float32)
        raw_p_train = pred_train["raw_p"].numpy()
        raw_p_test = pred_test["raw_p"].numpy()

        p_train = tf.math.sigmoid(raw_p_train) * (self.p_max(q) - self.p_min(q)) + self.p_min(q)
        p_test = tf.math.sigmoid(raw_p_test) * (self.p_max(q) - self.p_min(q)) + self.p_min(q)

        p_train = tf.clip_by_value(p_train, eps, 1.0-eps).numpy()
        p_test = tf.clip_by_value(p_test, eps, 1.0-eps).numpy()
    
        theta_train = self.C_inv( self.a0(q) / p_train, q )
        theta_test = self.C_inv( self.a0(q) / p_test, q )
    
        S0_ts_train = tf.cast( base_spec.survival(ts_grid, alpha_train), tf.float32 )
        S0_ts_test = tf.cast( base_spec.survival(ts_grid, alpha_test), tf.float32 )
        S0_train = tf.cast( base_spec.survival(y_train, alpha_train), tf.float32 )
        S0_test = tf.cast( base_spec.survival(y_test, alpha_test), tf.float32 )
        
        u_ts_train = S0_ts_train * self.phi(theta_train, q)
        u_ts_test = S0_ts_test * self.phi(theta_test, q)
        u_train = S0_train * self.phi(theta_train, q)
        u_test = S0_test * self.phi(theta_test, q)
        
        A_u_ts_train = self.A( u_ts_train, q )
        A_u_ts_test = self.A( u_ts_test, q )
        A_u_train = self.A( u_train, q )
        A_u_test = self.A( u_test, q )
    
        C_theta_train = self.C( theta_train, q )
        C_theta_test = self.C( theta_test, q )
        
        S_ts_train = A_u_ts_train / C_theta_train
        S_ts_test = A_u_ts_test / C_theta_test
        S_train = A_u_train / C_theta_train
        S_test = A_u_test / C_theta_test
        
        H_train = -np.log( S_train )
        H_test = -np.log( S_test )

        results_dict = {
            "ts_grid": ts_grid,
            "S_ts_train": S_ts_train,
            "S_ts_test": S_ts_test,
            "y_train": y_train,
            "y_test": y_test,
            "delta_train": delta_train,
            "delta_test": delta_test,
            "S_train": S_train,
            "S_test": S_test,
            "H_train": H_train,
            "H_test": H_test,
            "theta_train": theta_train,
            "theta_test": theta_test,
            "p_train": p_train,
            "p_test": p_test,
            "alpha_train": alpha_train,
            "alpha_test": alpha_test
        }
        # Runs through all elements in the results dictionary and convert them to numpy, if possible
        for key in results_dict:
            if(hasattr(results_dict[key], "to_numpy")):
                results_dict[key] = results_dict[key].to_numpy()
            if(hasattr(results_dict[key], "numpy")):
                results_dict[key] = results_dict[key].numpy()
        
        return results_dict

    model.get_survival_cure = types.MethodType( get_survival_cure, model )
    
    return model

B = 501

class BasePiecewiseExp:

    def __init__(self, n_cuts = 5, s = None, y = None, delta = None):
        self.n_parameters = n_cuts + 1
        if(s is None):
            if(y is None or delta is None):
                raise ValueError("Please, provide at least times and event indicators.")
            _, s = initialize_alpha_s(y, delta, n_cuts = n_cuts)
        self.s = s

        def pdf(y, alpha, force_broadcasting = False):
            return pwexp.pdf(y, alpha, self.s, force_broadcasting = force_broadcasting)

        def log_pdf(y, alpha, force_broadcasting = False):
            return tf.math.log( self.pdf(y,alpha, force_broadcasting = force_broadcasting) )
        
        def survival(y, alpha, force_broadcasting = False):
            return pwexp.cdf(y, alpha, self.s, lower_tail = False, force_broadcasting = force_broadcasting)
    
        def log_survival(y, alpha, force_broadcasting = False):
            return pwexp.log_survival(y, alpha, self.s, force_broadcasting = force_broadcasting)
    
        def h(y, alpha, force_broadcasting = False):
            return pwexp.h(y, alpha, self.s, force_broadcasting = force_broadcasting)
    
        def log_h(y, alpha, force_broadcasting = False):
            return tf.math.log( pwexp.h(y, alpha, self.s, force_broadcasting = force_broadcasting) )
    
        def log_f(y, alpha, force_broadcasting = False):
            log_S0 = pwexp.log_survival(y, alpha, self.s, force_broadcasting = force_broadcasting)
            log_h0 = tf.math.log( pwexp.h(y, alpha, self.s, force_broadcasting = force_broadcasting) )
            return log_h0 + log_S0
    
        self.survival = survival
        self.log_survival = log_survival
        self.h = h
        self.log_h = log_h
        self.log_f = log_f

class BaseWeibull:

    def __init__(self):
        self.n_parameters = 2

        def pdf(y, alpha):
            return tf.math.exp( self.log_pdf(y, alpha) )

        def log_pdf(y, alpha):
            k = alpha[0]
            lam = alpha[1]
            return tf.math.log(k) - k * tf.math.log(lam) + (k-1) * tf.math.log(y)
        
        def survival(y, alpha):
            return tf.math.exp( self.log_survival(y, alpha) )
    
        def log_survival(y, alpha):
            k = alpha[0]
            lam = alpha[1]
            return -(y / lam)**k
    
        def h(y, alpha):
            return tf.math.exp( self.log_h(y, alpha) )
    
        def log_h(y, alpha):
            k = alpha[0]
            lam = alpha[1]
            return tf.math.log(k) - k * tf.math.log(lam) + (k-1)*tf.math.log(y)
    
        def log_f(y, alpha):
            log_S0 = self.log_survival(y, alpha)
            log_h0 = self.log_h(y, alpha)
            return log_h0 + log_S0
    
        self.survival = survival
        self.log_survival = log_survival
        self.h = h
        self.log_h = log_h
        self.log_f = log_f

class MPSPoisson:

    def __init__(self):        
        self.hasq = False
        self.fixed_q = tf.cast(0.0, tf.float32)
        
        def log_a(m, q):
            return -tf.math.lgamma(m+1)

        def a(m, q):
            return tf.math.exp( self.log_a(m, q) )
    
        def a0(q):
            return tf.cast(1.0, tf.float32)
        
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

        def p_min(q):
            return tf.cast(0.0, tf.float32)

        def p_max(q):
            return tf.cast(1.0, tf.float32)

        def sup(q):
            return tf.cast( np.arange(B), tf.float32 )
        
        self.a = a
        self.a0 = a0
        self.log_a = log_a
        self.phi = phi
        self.log_phi = log_phi
        self.phi_inv = phi_inv
        self.C = C
        self.C_inv = C_inv
        self.A = A
        self.p_min = p_min
        self.p_max = p_max
        self.sup = sup


class MPSBinomial:

    def __init__(self, fixed_q):
        self.hasq = False
        self.fixed_q = tf.cast(fixed_q, tf.float32)
        
        def log_a(m, q):
            return tf.math.lgamma(q+1) - tf.math.lgamma(m+1) - tf.math.lgamma(q-m+1)

        def a(m, q):
            return tf.math.exp( log_a(m, q) )

        def a0(q):
            return tf.cast(1.0, tf.float32)
        
        def phi(theta, q):
            return theta / (1-theta)
        
        def log_phi(theta, q):
            return tf.math.log(theta) - tf.math.log(1-theta)
                               
        def phi_inv(u, q):
            return u / (1 + u)
        
        def C(theta, q):
            return (1-theta)**(-q)
        
        def C_inv(u, q):
            return 1 - u**(-1/q)
        
        def A(u, q):
            theta = phi_inv(u, q)
            return C(theta, q)

        def p_min(q):
            return tf.cast(0.0, tf.float32)

        def p_max(q):
            return tf.cast(1.0, tf.float32)

        def sup(q):
            return tf.cast( np.arange(q+1), tf.float32 )
        
        self.a = a
        self.a0 = a0
        self.log_a = log_a
        self.phi = phi
        self.log_phi = log_phi
        self.phi_inv = phi_inv
        self.C = C
        self.C_inv = C_inv
        self.A = A
        self.p_min = p_min
        self.p_max = p_max
        self.sup = sup

class MPSNegBinomial:

    def __init__(self, fixed_q = None):
        if(fixed_q is None):
            self.hasq = True
            self.fixed_q = tf.cast(0.0, tf.float32)
        else:
            self.hasq = False
            self.fixed_q = tf.cast(fixed_q, tf.float32)
        
        def log_a(m, q):
            return tf.math.lgamma(1/q+m) - tf.math.lgamma(1/q) - tf.math.lgamma(m+1)

        def a(m, q):
            return tf.math.exp( log_a(m, q) )

        def a0(q):
            return tf.cast(1.0, tf.float32)
        
        def phi(theta, q):
            return q * theta / (1 + q*theta)
        
        def log_phi(theta, q):
            return tf.math.log(q * theta) - tf.math.log(1 + q*theta)
                               
        def phi_inv(u, q):
            return u / (q * (1-u))
        
        def C(theta, q):
            return (1+q*theta)**(1/q)
        
        def C_inv(u, q):
            return (u**q - 1) / q
        
        def A(u, q):
            theta = phi_inv(u, q)
            return C(theta, q)

        def p_min(q):
            return tf.cast(0.0, tf.float32)

        def p_max(q):
            return tf.cast(1.0, tf.float32)

        def sup(q):
            sup = tf.cast( np.arange(B), tf.float32 )

        def link_q(q):
            return tf.math.softplus(q)
        
        def link_inv_q(u):
            # Inverse of the softplus function
            return tf.math.log(tf.math.exp(u) - 1)
        
        self.a = a
        self.a0 = a0
        self.log_a = log_a
        self.phi = phi
        self.log_phi = log_phi
        self.phi_inv = phi_inv
        self.C = C
        self.C_inv = C_inv
        self.A = A
        self.p_min = p_min
        self.p_max = p_max
        self.sup = sup
        self.link_q = link_q
        self.link_inv_q = link_inv_q

        
class MPSLogarithmic:

    def __init__(self):
        self.hasq = False
        self.fixed_q = tf.cast(0.0, tf.float32)
        
        def log_a(m, q):
            return -tf.math.log(m+1)

        def a(m, q):
            return tf.math.exp( log_a(m, q) )

        def a0(q):
            return tf.cast(1.0, tf.float32)
        
        def phi(theta, q):
            return tf.identity(theta)
        
        def log_phi(theta, q):
            return tf.math.log(theta)
                               
        def phi_inv(u, q):
            return tf.identity(u)
        
        def C(theta, q):
            # Identify small values of theta
            small_mask = tf.math.abs(theta) < 0.1
            
            # Just ensures standard will not produce NaNs over observations where Taylor expansion will be used
            safe_theta = tf.where(small_mask, tf.constant(0.1, dtype = theta.dtype), theta)
            standard = -tf.math.log(1 - safe_theta) / safe_theta

            # Obtain the Taylor expansion for values of theta near zero to avoid numerical autodiff problems
            taylor = 1.0 + theta / 2.0 + (theta**2) / 3.0 + (theta**3) / 4.0
            
            return tf.where(small_mask, taylor, standard)
            
            # return -tf.math.log(1-theta) / theta
        
        def C_inv(u, q):
            return 1 + tfp.math.lambertw(-u * tf.math.exp(-u)) / u
        
        def A(u, q):
            theta = phi_inv(u, q)
            return C(theta, q)

        def p_min(q):
            return tf.cast(0.0, tf.float32)

        def p_max(q):
            return tf.cast(1.0, tf.float32)

        def sup(q):
            return tf.cast(np.arange(B), tf.float32)
        
        self.a = a
        self.a0 = a0
        self.log_a = log_a
        self.phi = phi
        self.log_phi = log_phi
        self.phi_inv = phi_inv
        self.C = C
        self.C_inv = C_inv
        self.A = A
        self.p_min = p_min
        self.p_max = p_max
        self.sup = sup


class MPSRGP:

    def __init__(self, fixed_q = None):
        if(fixed_q is None):
            self.hasq = True
            self.fixed_q = tf.cast(0.0, tf.float32)
        else:
            self.hasq = False
            self.fixed_q = tf.cast(fixed_q, tf.float32)
        
        def log_a(m, q):
            return (m-1)*tf.math.log( 1 + q*m ) - tf.math.lgamma(m+1)

        def a(m, q):
            return tf.math.exp( log_a(m, q) )

        def a0(q):
            return tf.cast(1.0, tf.float32)
        
        def phi(theta, q):
            return theta * tf.math.exp(-q * theta)
        
        def log_phi(theta, q):
            return tf.math.log(theta) - q * theta
                               
        def phi_inv(u, q):
            return - tfp.math.lambertw( -q*u ) / q
        
        def C(theta, q):
            return tf.math.exp(theta)
        
        def C_inv(u, q):
            return tf.math.log(u)
        
        def A(u, q):
            theta = phi_inv(u, q)
            return C(theta, q)

        def p_min(q):
            return tf.cast(tf.math.exp(-tf.math.abs(1/q)), tf.float32)

        def p_max(q):
            return tf.cast(1.0, tf.float32)

        def sup(q):
            if(q > 0):
                return tf.cast( np.arange(B), tf.float32 )
            else:
                if(q < -1):
                    raise ValueError("q value can't be less than -1")
                max_sup = tf.math.ceil( tf.math.abs(1/q) ) - 1
                return np.arange(max_sup+1).astype(np.float64)

        def link_q(q):
            # We allow q to vary between -1/3 and infinity.
            # That allows us to obtain an RGP(q) model with support from {0,1,2}
            # At its extreme, q = -1/2 and the support of the RGP distribution collapses to {0,1} (i.e. a Bernoulli)
            # For q > -1/2, the support varies from {0,1,2}, {0,1,2,3}, ..., up to an infinite support when q >= 0
            # Specifically, fow q = 0, we recover the Poisson model
            return tf.math.softplus(q) - 1/2
        
        def link_inv_q(u):
            # Inverse of the softplus function translated 1/2 below
            return tf.math.log(tf.math.exp(u + 1/2) - 1)
            
        self.a = a
        self.a0 = a0
        self.log_a = log_a
        self.phi = phi
        self.log_phi = log_phi
        self.phi_inv = phi_inv
        self.C = C
        self.C_inv = C_inv
        self.A = A
        self.p_min = p_min
        self.p_max = p_max
        self.sup = sup
        self.link_q = link_q
        self.link_inv_q = link_inv_q














        