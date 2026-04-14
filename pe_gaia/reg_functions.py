import tensorflow as tf 
import numpy as np 


def modify_labels(X, y, mu, sigma):
    '''
    Function to obtain and standardize the desired labels for the dataset. 
    All written with tf function to parallelize the process. 
    ----------------------------------------------------------------------
    Arguments:
    X -- input time series
    y -- labels 
    mu -- array containing the labels mean of the train dataset 
    sigma -- array containing the labels std of the train dataset 
    
    Return:
    X -- input time series 
    new_y_standardized -- array containing the desired labels 
    '''
    m1, m2, z1, z2 = tf.unstack(y)

    m_sum = tf.add(m1, m2)
    m_prod = tf.multiply(m1, m2)

    chirp_mass_num = tf.pow(m_prod, 0.6)
    chirp_mass_den = tf.pow(m_sum, 0.2) 
    chirp_mass = tf.divide(chirp_mass_num, chirp_mass_den)
    
    total_mass = m_sum
    #q = tf.divide(m2, m1)
    
    term1 = tf.multiply(m1, z1)
    term2 = tf.multiply(m2, z2)
    effective_spin_num = tf.add(term1, term2)
    effective_spin = tf.divide(effective_spin_num, m_sum)
    
    new_y = tf.stack([chirp_mass, total_mass, effective_spin])

    diff = tf.subtract(new_y, mu)
    new_y_standardized = tf.divide(diff, sigma) 
    
    return X, new_y_standardized

def parse_example(example_proto):
    
    feature_spec = {
        "x": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
    }

    parsed = tf.io.parse_single_example(example_proto, feature_spec)

    x = tf.io.parse_tensor(parsed["x"], tf.float32)
    x = tf.ensure_shape(x, (16384, 3))

    y = tf.io.parse_tensor(parsed["y"], tf.float32)
    y = tf.ensure_shape(y, (12,))
    
    y = y[:4]
    y = tf.ensure_shape(y, (4,))

    return x, y


def create_tf_dataset(path):
    '''
    Function to create a tf dataset without loading the .npz file in memory. 
    '''
    
    npz = np.load(path, allow_pickle=True, mmap_mode='r') # does not load it in RAM
    X_map, y_map = npz['X'], npz['y']

    def gen():
        for i in range(len(X_map)):
            yield X_map[i], y_map[i, :4] # select only the desired labels (m1, m2, z1, z2)

    sig = (tf.TensorSpec(X_map.shape[1:] , tf.as_dtype(X_map.dtype)),
           tf.TensorSpec((4,), tf.as_dtype(y_map.dtype))) 
    
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    
    return ds

def moving_average(x, y, window_size):
    idx = np.argsort(x)
    x_s, y_s = x[idx], y[idx]
    y_avg = np.convolve(y_s, np.ones(window_size)/window_size, mode='valid')
    x_avg = np.convolve(x_s, np.ones(window_size)/window_size, mode='valid')
    return x_avg, y_avg