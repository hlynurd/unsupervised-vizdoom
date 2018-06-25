import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import glorot_normal
import numpy as np
import tensorflow as tf
import pickle
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

def sfa_vizdoom_default_cnn(dim1=30, dim2=45, available_actions_count=3):
    #tf.reset_default_graph() 
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True
    #set_session(tf.Session(config=config))
    net = Sequential()
    
    net.add(Conv2D(8, (6, 6), strides=3, input_shape=(dim1, dim2, 1), padding="same", activation="relu", kernel_initializer=glorot_normal(), bias_initializer=keras.initializers.Constant(value=0.1)))
    #net.add(Dropout(0.5))
    
    net.add(Conv2D(8, (3, 3), strides=2, padding="same", activation="relu", kernel_initializer=glorot_normal(), bias_initializer=keras.initializers.Constant(value=0.1)))
    #net.add(Dropout(0.5))
    
    net.add(Flatten())
    
    net.add(Dense(128, activation="relu", kernel_initializer=glorot_normal(), bias_initializer=keras.initializers.Constant(value=0.1)))    
    #net.add(Dropout(0.5))
    
    net.add(Dense(available_actions_count, kernel_initializer=glorot_normal(), bias_initializer=keras.initializers.Constant(value=0.1)))   
    optimizah = keras.optimizers.RMSprop(lr=0.00025, rho=0.0, epsilon=1e-10, decay=0.9)
    net.compile(loss='mean_squared_error', optimizer=optimizah)


    return net