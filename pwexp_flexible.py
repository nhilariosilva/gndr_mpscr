import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

# Hazard function
def h(t, alpha, s, force_broadcasting = False):
    """
        Hazard function for the piecewise exponential distribution.
        Supports broadcasting alpha.
    """
    alpha = tf.cast(alpha, tf.float64)
    s = tf.cast(s, tf.float64)

    if(alpha.shape[0] is not None):
        # If dimensions of alpha array and t exist and do not match, must use broadcasting
        if(t.shape[0] is not None and alpha.shape[0] != t.shape[0]):
            force_broadcasting = True

        # If alpha is a [1,k] array and t [n,1], broadcasting is always expected
        # If t is [1,1], then broadcasting would not be necessary, but as a standard implementation,
        # we can still consider broadcasting here, leading to a 2d array as result.
        # That implementation may also work when t is shape [None,1], as applied by tensorflow to map the operations in the Graph
        if(alpha.shape[0] == 1):
            force_broadcasting = True
    
    if(not force_broadcasting):
        t = tf.reshape(tf.cast(t, tf.float64), [-1])
        g = tf.searchsorted(s, t)
        g = tf.maximum(g, 1)
        h_t = tf.gather(alpha, g - 1, batch_dims=1)
        h_t = tf.reshape(h_t, [-1, 1])
    else:
        t = tf.reshape(tf.cast(t, tf.float64), [-1])
        g = tf.searchsorted(s, t)
        g = tf.maximum(g, 1)
        h_t = tf.gather(alpha, g - 1, axis=-1)

    return tf.cast(h_t, tf.float32)

# Cumulative Hazard
def ch(t, alpha, s, force_broadcasting = False):
    """
        Cumulative hazard function for the piecewise exponential distribution.
        Supports broadcasting alpha.
    """
    alpha = tf.cast(alpha, tf.float64)
    s = tf.cast(s, tf.float64)

    if(alpha.shape[0] is not None):
        # If dimensions of alpha array and t exist and do not match, must use broadcasting
        if(t.shape[0] is not None and alpha.shape[0] != t.shape[0]):
            force_broadcasting = True

        # If alpha is a [1,k] array and t [n,1], broadcasting is always expected
        # If t is [1,1], then broadcasting would not be necessary, but as a standard implementation,
        # we can still consider broadcasting here, leading to a 2d array as result.
        # That implementation may also work when t is shape [None,1], as applied by tensorflow to map the operations in the Graph
        if(alpha.shape[0] == 1):
            force_broadcasting = True

    # Pre-calculate the areas under each piecewise section
    interval_widths = s[1:] - s[:-1]
    hazard_increments = alpha[:, :-1] * interval_widths
    
    # Accumulate the lags along the interval axis (axis=1)
    N_batch = tf.shape(alpha)[0]
    zeros = tf.zeros([N_batch, 1], dtype=tf.float64)
    all_lags = tf.concat([zeros, tf.math.cumsum(hazard_increments, axis=1)], axis=1)

    # Flatten times to map the indices
    t_flat = tf.reshape(tf.cast(t, tf.float64), [-1])
    g = tf.searchsorted(s, t_flat)
    g = tf.maximum(g, 1)

    # The knots 's' are just 1D, so gather always extracts a 1D tensor
    respective_s = tf.gather(s, g - 1)

    if(not force_broadcasting):
        cummulated_lags = tf.gather(all_lags, g - 1, batch_dims=1)
        respective_alpha = tf.gather(alpha, g - 1, batch_dims=1)
        
        H_t = respective_alpha * (t_flat - respective_s) + cummulated_lags
        H_t = tf.reshape(H_t, [-1, 1]) 
    else:
        cummulated_lags = tf.gather(all_lags, g - 1, axis=-1)
        respective_alpha = tf.gather(alpha, g - 1, axis=-1)
        H_t = respective_alpha * (t_flat - respective_s) + cummulated_lags

    return tf.cast(H_t, tf.float32)

# Distribution function
def cdf(t, alpha, s, lower_tail = True, force_broadcasting = False):
    """
        Cumulative density function for the piecewise exponential distribution.
        Supports broadcasting alpha.
    """
    S = tf.math.exp( -ch(t, alpha, s, force_broadcasting = force_broadcasting) )
    if(lower_tail):
        return( 1 - S )
    return S

def log_survival(t, alpha, s, force_broadcasting = False):
    """
        Log survival function for the piecewise exponential distribution.
        Supports broadcasting alpha.
    """
    
    log_S = -ch(t, alpha, s, force_broadcasting = force_broadcasting)
    return log_S

# Probability density function
def pdf(t, alpha, s, force_broadcasting = False):
    """
        Density function for the piecewise exponential distribution.
        Supports broadcasting alpha.
    """
    return h(t, alpha, s) * cdf(t, alpha, s, lower_tail = False, force_broadcasting = force_broadcasting)

# Quantile function - Assume alpha is a single vector [1, s.shape[0]] (function not vectorized)
def ppf(q, alpha, s):
    """
        Quantile function for the piecewise exponential distribution.
        Does not support broadcasting for alpha. That means it will treat alpha to be a constant vector
    """
    # alpha is assumed to be a single vector
    alpha_single_vector = alpha[0,:]
    
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
    hazard_increments = alpha_single_vector[:-1] * interval_widths
    # g=1 has 0 accumulated lag, g=2 has alpha_1 (s_1 - s_0), and so on.
    all_lags = tf.concat([tf.constant([0.0], dtype=tf.float64), tf.math.cumsum(hazard_increments)], axis = 0)
    # For each quantile, obtain its correponding cummulated hazard depending on its respective position according to t
    cummulated_lags = tf.gather(all_lags, g-1)

    # For each t, obtain which alpha corresponds to it
    respective_alpha = tf.gather(alpha_single_vector, g-1)
    # For each t, obtain which knot is directly below it
    respective_s = tf.gather(s, g-1)
    
    F_inv = -(tf.math.log(1-q) + cummulated_lags) / respective_alpha + respective_s

    F_inv = tf.reshape(F_inv, original_shape)
    return tf.cast(F_inv, tf.float32)

def rvs(alpha, s, size = 1):
    """
        Sampler for the piecewise exponential distribution.
        Does not support broadcasting for alpha. That means it will treat alpha to be a constant vector
    """
    u = tf.random.uniform(minval = 0, maxval = 1, shape = (size,))
    return ppf(u, alpha, s)
