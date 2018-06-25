import keras
from keras.models import Sequential
from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import pickle
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

def sfa_vizdoom_default_cnn(dim1=30, dim2=45, output_features=3):
    tf.reset_default_graph() 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))
    net = Sequential()
    
    net.add(Conv2D(8, (6, 6), strides=3, input_shape=(dim1, dim2, 1), padding="same", activation="relu"))
    net.add(BatchNormalization())
    net.add(Dropout(0.5))
    
    net.add(Conv2D(8, (3, 3), strides=2, padding="same", activation="relu"))
    net.add(BatchNormalization())
    net.add(Dropout(0.5))
    
    net.add(Flatten())
    
    net.add(Dense(128, activation="relu"))    
    net.add(BatchNormalization())
    net.add(Dropout(0.5))
    
    net.add(Dense(output_features))    
    net.add(PowerWhitening(output_dim=output_features, n_iterations=50))    
    net.compile(loss=sfa_loss, optimizer='adam')
    
    return net

def instantiate_light_cnn(dim1=32, dim2=32, output_features=3, supervised=False):
    tf.reset_default_graph() 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))
    net = Sequential()
    
    net.add(Conv2D(18, (7, 7), input_shape=(dim1, dim2, 1), padding="same", activation="relu"))
    net.add(BatchNormalization())
    net.add(Dropout(0.5))
    net.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    
    net.add(Conv2D(18, (5, 5), padding="same", activation="relu"))
    net.add(BatchNormalization())
    net.add(Dropout(0.5))
    net.add(MaxPooling2D(pool_size=(2, 2)))
   
    net.add(Conv2D(1, (1, 1), padding="same", activation="relu"))
    net.add(BatchNormalization())
    net.add(Dropout(0.5))
    net.add(Flatten())
    
    net.add(Dense(output_features))    
    
    
    if not supervised:
        net.add(PowerWhitening(output_dim=output_features, n_iterations=50))    
        net.compile(loss=sfa_loss, optimizer='adam')
    else:
        net.add(Activation('softmax'))    
        net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return net

def general_sfa_net(dim1=32, dim2=32, channels=1, output_features=3, supervised=False, generalized_sfa=True):
    tf.reset_default_graph() 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))
    net = Sequential()
    
    net.add(Conv2D(16, (2, 2), input_shape=(dim1, dim2, channels), padding="same", activation="relu"))
    net.add(BatchNormalization())
    #net.add(Dropout(0.5))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    net.add(Conv2D(16, (2, 2), padding="same", activation="relu"))
    net.add(BatchNormalization())
    #net.add(Dropout(0.5))
    net.add(MaxPooling2D(pool_size=(2, 2)))


    net.add(Flatten())
    net.add(Dense(1000))
    net.add(Activation("relu"))
#    net.add(BatchNormalization())

    net.add(Dense(output_features))
    net.add(PowerWhitening(output_dim=output_features, n_iterations=50))    
    #nadam = keras.optimizers.Nadam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #rmsprop = keras.optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=None, decay=0.0)
    if generalized_sfa:
        net.compile(loss=generalized_sfa_loss, optimizer='nadam')
    else:
        net.compile(loss=sfa_loss, optimizer='nadam')
    return net

def mobile_net(dim1=32, dim2=32, output_features=3):
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))
    from keras.applications.mobilenet import MobileNet
    from keras.models import Model
    mobi = MobileNet(input_shape=(96, 96, 2), weights=None)
    pre_output = mobi.get_layer("dropout").output
    pre_output = Flatten()(pre_output)
    pre_output = keras.layers.Dense(1000)(pre_output)
    pre_output = keras.layers.Dense(output_features)(pre_output)
    pre_output = PowerWhitening(output_dim=output_features, n_iterations=50)(pre_output)
    net = Model(input=mobi.input, output=pre_output)
    net.compile(loss=generalized_sfa_loss, optimizer='nadam')
    return net

def nas_net(dim1=32, dim2=32, output_features=3):
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))
    from keras.applications.nasnet import NASNetMobile
    from keras.models import Model
    mobi = NASNetMobile(input_shape=(96, 96, 2), weights=None)
    pre_output = mobi.layers[-2].output
    pre_output = keras.layers.Dense(output_features)(pre_output)
    pre_output = PowerWhitening(output_dim=output_features, n_iterations=50)(pre_output)
    net = Model(input=mobi.input, output=pre_output)
    net.compile(loss=generalized_sfa_loss, optimizer='nadam')
    return net

def instantiate_smooth_cnn(dim1=96, dim2=96, output_features=3, supervised=False):
    tf.reset_default_graph() 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))
    net = Sequential()
    
    net.add(Conv2D(18, (7, 7), input_shape=(dim1, dim2, 2), padding="same"))
    net.add(BatchNormalization())
    net.add(Activation('relu'))
    net.add(Dropout(0.5))
    net.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    
    net.add(Conv2D(24, (5, 5), padding="same"))
    net.add(BatchNormalization())
    net.add(Activation('relu'))
    net.add(Dropout(0.5))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    
    net.add(Conv2D(24, (5, 5), padding="same"))
    net.add(BatchNormalization())
    net.add(Activation('relu'))
    net.add(Dropout(0.5))
    net.add(MaxPooling2D(pool_size=(2, 2)))
    
    net.add(Conv2D(24, (3, 3), padding="same"))
    net.add(BatchNormalization())
    net.add(Activation('relu'))
    net.add(Dropout(0.5))
    net.add(MaxPooling2D(pool_size=(2, 2)))
   
    net.add(Conv2D(1, (1, 1), padding="same"))
    net.add(BatchNormalization())
    net.add(Dropout(0.5))
    net.add(Flatten())
    net.add(Dense(output_features))    
    
    if not supervised:
        net.add(PowerWhitening(output_dim=output_features, n_iterations=50))

        rmsprop = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        net.compile(loss=generalized_sfa_loss, optimizer="nadam")
    else:
        net.add(activation('softmax'))
    
        rmsprop = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
        net.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['categorical_accuracy'])
    return net

def matrix_power(A, x, power, eigenvalue_computation="complex"):
    with tf.name_scope("matrix_power"):
        for _ in range(power):
            x = tf.matmul(A, x)
            x = tf.div(x, tf.norm(x))
        if eigenvalue_computation == "simple":
            e = tf.div(tf.matmul(A, x)[0], x[0])
        else:
            e = tf.norm(tf.matmul(A, x))
    return x, e

class PowerWhitening(Layer):
    
    def __init__(self, output_dim=1, n_iterations=100, distribution="uniform", **kwargs):
        self.output_dim = output_dim
        self.n_iterations = n_iterations
        self.distribution = distribution
        super(PowerWhitening, self).__init__(**kwargs)        
        
    def build(self, input_shape):
        super(PowerWhitening, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        
    def get_output_shape_for(self, input_shape):
        return (input_shape)

    def call(self, input_tensor, mask=None):
        with tf.name_scope("power_whitening"):
            if self.distribution == "uniform":
                iterations = [self.n_iterations] * self.output_dim
            elif self.distribution == "position_relative":
                iterations = [i * self.n_iterations for i in range(1, self.output_dim+1)]
            elif self.distribution is list:
                iterations = distribution
            elif self.distribution == "increment":
                increment = kwargs.get("increment", 10)
                iterations = [self.n_iterations + i * increment for i in range(0, self.output_dim)]
            R = tf.get_variable("randomwhiteningvectors",
                                initializer= np.random.normal(size=(self.output_dim, self.output_dim)).astype(np.float32),
                                trainable = False)
            approx_W = tf.get_variable("whiteningmatrix",
                                       initializer=np.zeros(shape=(self.output_dim, self.output_dim)).astype(np.float32),
                                       trainable=False)
            input_mean, _ = tf.nn.moments(input_tensor, axes=[0])
            input_tensor = input_tensor - input_mean[None, :]
            C = tf.div(tf.matmul(input_tensor, input_tensor, True, False), tf.cast(tf.shape(input_tensor)[0], tf.float32))
            covariance_matrix = C
            for i in range(self.output_dim):
                evector, evalue = matrix_power(C, R[:, i, None], iterations[i])
                C = C - evalue * tf.matmul(evector, evector, False, True)
                approx_W += 1 / tf.sqrt(evalue) * tf.matmul(evector, evector, False, True)
            whitened_output = tf.matmul(input_tensor, approx_W, False, True)
        return whitened_output#, approx_W, input_mean, covariance_matrix
    
    
'''
def sfa_loss(input_tensor):
    output_dim = input_tensor.shape[1]
    feature_weight = tf.linspace(1., 0.3, output_dim)
    #feature_weight = tf.constant(1.)
    weighted_whitened_output = input_tensor * feature_weight
    difference_squared_norms = tf.reduce_sum(tf.squared_difference(weighted_whitened_output[1:],
                                                                   weighted_whitened_output[:-1]),
                                             axis=1)
    loss = tf.reduce_mean(difference_squared_norms, axis=0)
    return loss
'''


#def sfa_loss(yTrue, yPred):
#    output_dim = yPred.shape[1]
#    w = tf.linspace(1., 0.3, output_dim)
#    #w = tf.constant(1.)
#    weighted_whitened_output = yPred * w
#    loss = tf.reduce_mean(tf.losses.mean_squared_error(weighted_whitened_output[1:], weighted_whitened_output[:-1]))
#    return loss

def sfa_loss(yTrue, yPred):
    output_dim = yPred.shape[1]
    print(yPred.shape)
    w = tf.linspace(1., 0.3, output_dim)
    w = w / tf.reduce_sum(w)
    #w = tf.constant(1.)
    squared_differences = tf.squared_difference(yPred[1:], yPred[:-1])
    weighted_differences = w * squared_differences
    avg_slowness_per_feature = tf.reduce_mean(weighted_differences, axis=0)
    avg_slowness = tf.reduce_mean(avg_slowness_per_feature, axis=0)
    return avg_slowness

def generalized_sfa_loss(yTrue, yPred):
    """
    Weight tensor should be a matrix that contains the weights for each part
    """
    # this needs to be a bit hacky to easily comply with the keras mold
    weight_matrix_tf = yTrue
    input_tensor = yPred
    output_dim = input_tensor.shape[1]
    feature_weight = tf.linspace(1., 0.3, output_dim)

    laplacian_tf = compute_normalized_laplacian(weight_matrix_tf)
    weighted_whitened_output = input_tensor * feature_weight
    auxiliary_matrix = tf.matmul(weighted_whitened_output, tf.matmul(laplacian_tf, weighted_whitened_output), True)
    auxiliary_diagonal = tf.diag_part(auxiliary_matrix)
    loss = tf.reduce_mean(auxiliary_diagonal)
    return loss


def compute_unnormalized_laplacian(weight_matrix_tf):
    """
    (Tensorflow)
    Provides a computation graph for the unnormalized Laplacian matrix as L = D - W
    from the weight matrix W.
    :param weight_matrix_tf: the tensor for the weight matrix W - should be real & symmetric
    :return: tensor for the unnormalized Laplacian matrix, shape=(n_data, n_data).
    """
    degree_vector_tf = tf.reduce_sum(weight_matrix_tf, 0)
    degree_matrix_tf = tf.diag(degree_vector_tf)
    laplacian_tf = degree_matrix_tf - weight_matrix_tf
    return laplacian_tf

def compute_normalized_laplacian(weight_matrix_tf):
    """
    (Tensorflow)
    Provides a computation graph for the normalized Laplacian matrix as L = I - D^(-1/2)WD^(-1/2)
    from the weight matrix W.
    :param weight_matrix_tf: the tensor for the weight matrix W - should be real & symmetric
    :return: tensor for the normalized Laplacian matrix, shape=(n_data, n_data).
    """
    degree_vector_tf = tf.reduce_sum(weight_matrix_tf, 0)
    degree_matrix_tf = tf.diag(degree_vector_tf)
    laplacian_tf = degree_matrix_tf - weight_matrix_tf
    sqrt_inv_degree_matrix_tf = tf.diag(1./tf.sqrt(degree_vector_tf))
    normalized_laplacian_tf = tf.matmul(sqrt_inv_degree_matrix_tf, tf.matmul(laplacian_tf, sqrt_inv_degree_matrix_tf))
    return normalized_laplacian_tf


def average_validation_delta(validation_numbers, dim1, dim2, net, sorting):
    average_deltas = 0
    for number in validation_numbers:
        current_data = prepare_image_sequence(number, dim1, dim2, 'train', sorting)
        average_deltas += average_delta(net.predict(current_data))
    return average_deltas / float(len(validation_numbers))
    

def average_delta(prediction):
    average_delta = 0
    lowest_delta = float('inf')
    for i in range(prediction.shape[1]):
        feature_i_values = prediction[:,i]
        
        delta = 0
        for j in range(len(feature_i_values)-1):
            addition = np.power((feature_i_values[j] - feature_i_values[j+1]),2)
            delta += addition
        delta /= float(len(feature_i_values))
        if delta < lowest_delta:
            lowest_delta = delta
        average_delta += delta
    average_delta = average_delta/float(prediction.shape[1])
    return average_delta


def print_deltas(prediction):
    average_delta = 0
    lowest_delta = float('inf')
    for i in range(prediction.shape[1]):
        feature_i_values = prediction[:,i]
        
        delta = 0
        for j in range(len(feature_i_values)-1):
            addition = np.power((feature_i_values[j] - feature_i_values[j+1]),2)
            delta += addition
        delta /= float(len(feature_i_values))
        if delta < lowest_delta:
            lowest_delta = delta
        average_delta += delta
        print("feature %i, delta: %.3f" %(i, delta))
    print("average delta value: %.2f" % (average_delta/float(prediction.shape[1])))
    print("lowest delta value: %.2f" % (lowest_delta))
    
