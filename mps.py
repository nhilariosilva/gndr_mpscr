import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

def pmf(x, log_a, log_phi, theta, q, sup, force_broadcasting = False):
    x = tf.cast(x, tf.float64)
    x_shape = tf.shape(x)
    x = tf.squeeze( x )
    
    theta = tf.squeeze( tf.cast(theta, tf.float64) )
    theta_col = tf.reshape( theta, (-1,1) )
    q = tf.squeeze( tf.cast(q, tf.float64) )
    q_col = tf.reshape( q, (-1,1) )
    sup = tf.cast(sup, tf.float64)

    is_scalar = (tf.rank(theta) == 0 or tf.shape(theta)[0] == 1) and (tf.rank(q) == 0 or tf.shape(q)[0] == 1)
    
    # Get the kernel values for the support of the distribution
    log_Psup_ker = log_a(sup, q) + sup * log_phi(theta_col, q_col)
    # Get the kernel values for the x to be evaluated in the distribution
    log_Px_ker = log_a(x, q) + x * log_phi(theta_col, q_col)
    
    # Z = sum_x Psup - In order to stabilize the sum of exponentials numerically, we use the LogSumExp trick
    # log_Z = log(exp( log_Psup1 + log_Psup2 + ... ))
    log_Z = tf.math.reduce_logsumexp(log_Psup_ker, axis = 1, keepdims = True)

    log_Px = log_Px_ker - log_Z
    Px = tf.math.exp(log_Px)
    
    if(tf.size(x) == tf.size(theta) and not force_broadcasting):
        Px = tf.linalg.diag_part(Px)
        if( tf.size(theta) == 1 ):
            Px = tf.reshape(Px, x_shape)
    
    return tf.cast(Px, tf.float32)

def rvs(log_a, log_phi, theta, q, sup, size = 1):
    theta = tf.cast(theta, tf.float64)
    q = tf.cast(q, tf.float64)
    sup = tf.cast(sup, tf.float64)
    
    # Check if theta and q are scalar i.e. a single constant value
    is_scalar = (tf.rank(theta) == 0 or tf.shape(theta)[0] == 1) and (tf.rank(q) == 0 or tf.shape(q)[0] == 1)
    # If theta is a scalar or a single valued array, the sample size is simply the value requested
    if is_scalar:
        theta = tf.reshape(theta, [1])
        num_samples = size
    # If theta is properly a vector, the size variable is not used, we sample exactly the size of theta (one sample for each theta)
    else:
        # No seu código original, se theta é vetor, size não é usado (1 amostra por theta)
        num_samples = 1

    theta_col = tf.reshape(theta, [-1, 1])
    q_col = tf.reshape(theta, [-1, 1])
    
    # Pass theta to be a column vector
    theta_col = tf.reshape(theta, [-1, 1])
    # Obtain the log_P for all values in the support
    log_Psup_ker = log_a(sup, q) + sup * log_phi(theta_col, q)
    # Sample the indices considering log_Psup_ker as weights
    indices = tf.random.categorical(log_Psup_ker, num_samples = size)
    sample = tf.squeeze( tf.gather(sup, indices) )
    
    return tf.cast(sample, tf.float32)