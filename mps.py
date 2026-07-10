import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

def pmf(x, log_a, log_phi, theta, q, sup, force_broadcasting = False):
    """
        Probability mass for an arbitrary, modified power series discrete distribution.
        Here, we assume that theta may vary, but q is always a fixed constant for simulation.
    """
    
    x = tf.cast(x, tf.float64)
    x_shape = tf.shape(x)
    x = tf.squeeze( x )
    
    theta = tf.squeeze( tf.cast(theta, tf.float64) )
    theta_col = tf.reshape( theta, (-1,1) )
    q = tf.squeeze( tf.cast(q, tf.float64) )
    sup = tf.cast(sup, tf.float64)

    theta_scalar = (tf.rank(theta) == 0 or tf.shape(theta)[0] == 1)
    
    # Get the kernel values for the support of the distribution
    log_Psup_ker = log_a(sup, q) + sup * log_phi(theta_col, q)
    # Get the kernel values for the x to be evaluated in the distribution
    log_Px_ker = log_a(x, q) + x * log_phi(theta_col, q)
    
    # Z = sum_x Psup - In order to stabilize the sum of exponentials numerically, we use the LogSumExp trick
    # log_Z = log(exp( log_Psup1 + log_Psup2 + ... ))
    log_Z = tf.math.reduce_logsumexp(log_Psup_ker, axis = 1, keepdims = True)

    log_Px = log_Px_ker - log_Z
    Px = tf.math.exp(log_Px)
    
    if(tf.size(x) == tf.size(theta) and not force_broadcasting):
        Px = tf.linalg.diag_part(Px)

    # If theta is a scalar, keep the sample as a simple array
    if(theta_scalar):
        Px = tf.squeeze( Px )
    
    return tf.cast(Px, tf.float32)

def rvs(log_a, log_phi, theta, q, sup, size = 1):
    theta_scalar = (tf.rank(theta) == 0 or tf.shape(theta)[0] == 1)

    indices = np.arange( int(tf.size(sup)) )
    # If theta is a scalar, take a sample of requested size considering the probabilities for each value of the support
    if(theta_scalar):
        Psup = tf.cast( pmf(sup, log_a, log_phi, theta, q, sup), tf.float64 )
        # Just to correct numerical errors, ensure that the probabilities must sum to exactly one
        Psup = Psup / np.sum(Psup)
        sample_indices = np.random.choice(indices, size = size, p = Psup)
    # If theta is a vector, always assume we want a single realization for each value of theta.
    # That involves obtaining the probabilities of the support for each theta, which motivates a mandatory broadcasting
    # Just in case the user passes a theta vector the same size as the support, we ensure the sampler will act consistently
    else:
        Psup = tf.cast( pmf(sup, log_a, log_phi, theta, q, sup, force_broadcasting = True), tf.float64 )
        # Just to correct numerical errors, ensure that the probabilities must sum to exactly one
        Psup = Psup / np.sum(Psup, axis = 1, keepdims = True)
        sample_indices = np.array([
            np.random.choice(indices, size = 1, p = Psup[i,:])
            for i in range( int(tf.size(theta)) )
        ])
        sample_indices = tf.squeeze( sample_indices )

    sample = tf.gather(sup, sample_indices)
    
    return tf.cast(sample, tf.float32)