import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from esn_red import MiniESN

def esn_matrix_extract(model):
    # this function extracts the matrices of an ESN network
    W = tf.transpose(model.layers[0].weights[0])
    W_in = tf.transpose(model.layers[0].weights[1])

    W_out = tf.transpose(model.layers[1].weights[0])
    out_bias = tf.transpose(model.layers[1].weights[1])
    out_bias = tf.reshape(out_bias, [W_out.shape[0], 1])
    
    return W, W_in, W_out, out_bias

def esn_ss_sim(W, W_in, W_out, out_bias, leaky_ratio, inputs):
    # simulate the ESN state space model
    num_units = W.shape[0]
    num_outputs = W_out.shape[0]
    x_pre = np.zeros((num_units,1)) # initiate state as zeros if esn model use default zero initial state
    g_sample_all = np.zeros((num_units, inputs.shape[1])) # store all the activation function (function g() in paper) values, will be used as samples for MOR
    x_sample_all = np.zeros((num_units, inputs.shape[1])) # store all the states, will be used as samples for training and MOR
    y_out = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
    for i in range(inputs[0].shape[0]):    
        g_sample = leaky_ratio*np.tanh(W@x_pre + W_in@tf.reshape(inputs[0,i,:],[num_outputs,1]))
        x_cur = (1-leaky_ratio)*x_pre + g_sample
        y_out[:,[i]] = W_out @ x_cur + out_bias
        g_sample_all[:,[i]] = g_sample # record current activation function (function g() in paper) values as samples for MOR later
        x_sample_all[:,[i]] = x_cur # record current state in all state vector as samples for training and MOR
        x_pre = x_cur
    return y_out, g_sample_all, x_sample_all

def esn_red_sim(W, W_in, W_out_r, out_bias, V, leaky_ratio, inputs):
    # simulate the reduced ESN state space model without DEIM
    print("** simulating the reduced model...")
    order = V.shape[1]
    num_outputs = W_out_r.shape[0]
    x_pre_r = np.zeros((order,1)) # initiate state as zeros if esn model use default zero initial state
    y_out_r = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
    # print("shape_inputs: ", inputs.shape)
    for i in range(inputs[0].shape[0]):    
        x_cur_r = (1-leaky_ratio)*x_pre_r + leaky_ratio*V.T@np.tanh(W@V@x_pre_r + W_in@tf.reshape(inputs[0,i,:],[num_outputs,1]))
        y_out_r[:,[i]] = W_out_r @ x_cur_r + out_bias
        x_pre_r = x_cur_r
    return y_out_r

def esn_deim_sim(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, inputs):
    # simulate the reduced ESN model with DEIM
    print("** simulating the DEIM reduced model...")
    order = W_deim.shape[0]
    num_outputs = W_out_deim.shape[0]
    x_pre_deim = np.zeros((order,1)) # initiate state as zeros if esn model use default zero initial state
    y_out_deim = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
    x_sample_deim_all = np.zeros((order, inputs.shape[1])) # store all the states, will be used as samples for training
    for i in range(inputs[0].shape[0]):    
        x_cur_deim = (1-leaky_ratio)*x_pre_deim + leaky_ratio*E_deim@np.tanh(W_deim@x_pre_deim + W_in_deim@tf.reshape(inputs[0,i,:],[num_outputs,1]))
        x_sample_deim_all[:,[i]] = x_cur_deim # record current state in all state vector as samples for training
        y_out_deim[:,[i]] = W_out_deim @ x_cur_deim + out_bias
        x_pre_deim = x_cur_deim
    return y_out_deim, x_sample_deim_all

def esn_deim_assign(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, stime):
    # create reduced ESN network and assign weights
    print("** creating reduced ESN network and assigning weights...")
    order = W_deim.shape[0]
    num_inputs = W_in_deim.shape[1]
    num_outputs = W_out_deim.shape[0]
    recurrent_layer_red =  MiniESN(units=order, leaky=leaky_ratio, activation='tanh', connectivity=1, input_shape=(stime, num_inputs), return_sequences=True, use_bias=False, name="nn")
    output_red = keras.layers.Dense(num_outputs, name="readouts")
    # put all together in a keras sequential model
    model_red = keras.models.Sequential()
    model_red.add(recurrent_layer_red)
    model_red.add(output_red)
    model_red.summary()
    model_red.layers[0].weights[0].assign(tf.transpose(W_deim))
    model_red.layers[0].weights[1].assign(tf.transpose(W_in_deim))
    model_red.layers[0].weights[2].assign(tf.transpose(E_deim))
    model_red.layers[1].weights[0].assign(tf.transpose(W_out_deim))
    out_bias = tf.reshape(out_bias, [num_outputs])
    model_red.layers[1].weights[1].assign(tf.transpose(out_bias))
    return model_red

def esn_train(x_all, y_train):
    # num_outputs = W_out.shape[0]
    # num_units = W_out.shape[1]
    # n_t = x_all.shape[1]
    print("x_all_shape: ", x_all.shape)
    print("y_train_shape: ", y_train.shape)
    W_out = np.linalg.solve(x_all@(x_all.T),x_all@(y_train.T)).T
    W_out = W_out.astype('float32') # convert to float to be compatible with tensorflow
    # W_out = np.linalg.solve(x_all.T,y_train.T).T
    return W_out

def esn_assign(model, W_out):
    print("** assigning the trained weights to ESN...")
    model.layers[1].weights[0].assign(tf.transpose(W_out))
    return model
