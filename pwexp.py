import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

# Hazard function
def h(t, alpha, s):
    '''
        Hazard function for the piecewise exponential distribution.
        We assume s to be a vector of knots, whose first position is always equal to zero, that is s = [s_0, s_1, ..., s_k], with s_0 = 0.0
    '''
    original_shape = tf.shape(t)
    t = tf.cast(t, tf.float64)
    t = tf.squeeze(t)
    alpha = tf.cast(alpha, tf.float64)
    s = tf.cast(s, tf.float64)
    g = tf.searchsorted(s, t)
    # If any time is less or equal to zero, clip it to be slightly higher than 0.0, assuming the risk from (s_0, s_1]
    g = tf.maximum(g, 1)
    # Gather the risks for each corresponding time
    h_t = tf.gather(alpha, g-1)
    
    h_t = tf.reshape(h_t, original_shape)
    return tf.cast( h_t, tf.float32 )

def ch(t, alpha, s):
    '''
        Cumulative hazard function for the piecewise exponential distribution. 
        The integral of h from 0 to t.
        We assume s to be a vector of knots, whose first position is always equal to zero, that is s = [0.0, ...]
    '''
    original_shape = tf.shape(t)
    t = tf.cast(t, tf.float64)
    t = tf.squeeze(t)
    alpha = tf.cast(alpha, tf.float64)
    s = tf.cast(s, tf.float64)
    g = tf.searchsorted(s, t)
    # If any time is less or equal to zero, clip it to be slightly higher than 0.0, assuming the risk from (s_0, s_1]
    g = tf.maximum(g, 1)
    # Obtain the intervals between knots
    interval_widths = s[1:] - s[:-1]
    # Pre-calculate the areas under each piecewise section of the hazard function
    hazard_increments = alpha[:-1] * interval_widths
    # g=1 has 0 accumulated lag, g=2 has alpha_1 (s_1 - s_0), and so on.
    all_lags = tf.concat([tf.constant([0.0], dtype=tf.float64), tf.math.cumsum(hazard_increments)], axis = 0)
    # For each time, obtain its correponding cummulated hazard depending on its respective position according to t
    cummulated_lags = tf.gather(all_lags, g-1)
    # For each t, obtain which alpha corresponds to it
    respective_alpha = tf.gather(alpha, g-1)
    # For each t, obtain which knot is directly below it
    respective_s = tf.gather(s, g-1)
    # Sum the cummulated risk in the current block of knots with the accumulated risk from all previous blocks (vectorized)
    H_t = respective_alpha * (t - respective_s) + cummulated_lags
    
    H_t = tf.reshape(H_t, original_shape)
    return tf.cast( H_t, tf.float32 )

# Função de distribuição
def cdf(t, alpha, s, lower_tail = True):
    S = tf.math.exp( -ch(t, alpha, s) )
    if(lower_tail):
        return( 1 - S )
    return S

def log_survival(t, alpha, s):
    log_S = -ch(t, alpha, s)
    return log_S

# Função densidade de probabilidade
def pdf(t, alpha, s):
    return h(t, alpha, s) * cdf(t, alpha, s, lower_tail = False)

# Função quantil
def ppf(q, alpha, s):
    original_shape = tf.shape(q)
    q = tf.cast(q, tf.float64)
    q = tf.squeeze(q)
    
    alpha = tf.cast(alpha, tf.float64)
    s = tf.cast(s, tf.float64)

    s_inv = cdf(s, alpha, s)
    s_inv = tf.cast(s_inv, tf.float64)

    # Instead of searching with respect to time, uses the quantile values in the inverse knots (from 0 to 1)
    g = tf.searchsorted(s_inv, q)
    g = tf.maximum(g, 1)
    # Obtain the intervals between knots
    interval_widths = s[1:] - s[:-1]
    # Pre-calculate the areas under each piecewise section of the hazard function
    hazard_increments = alpha[:-1] * interval_widths
    # g=1 has 0 accumulated lag, g=2 has alpha_1 (s_1 - s_0), and so on.
    all_lags = tf.concat([tf.constant([0.0], dtype=tf.float64), tf.math.cumsum(hazard_increments)], axis = 0)
    # For each quantile, obtain its correponding cummulated hazard depending on its respective position according to t
    cummulated_lags = tf.gather(all_lags, g-1)

    # For each t, obtain which alpha corresponds to it
    respective_alpha = tf.gather(alpha, g-1)
    # For each t, obtain which knot is directly below it
    respective_s = tf.gather(s, g-1)
    
    F_inv = -(tf.math.log(1-q) + cummulated_lags) / respective_alpha + respective_s

    F_inv = tf.reshape(F_inv, original_shape)
    return tf.cast(F_inv, tf.float32)

# Função de amostragem
def rvs(alpha, s, size = 1):
    u = tf.random.uniform(minval = 0, maxval = 1, shape = (size,))
    return ppf(u, alpha, s)
