import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from esn_red import miniESN_unstable, miniESN_stable
import time
from scipy import sparse

def esn_matrix_extract(model):
    # this function extracts the matrices of an ESN network
    W = tf.transpose(model.layers[0].weights[0])
    W_in = tf.transpose(model.layers[0].weights[1])

    W_out = tf.transpose(model.layers[1].weights[0])
    out_bias = tf.transpose(model.layers[1].weights[1])
    out_bias = tf.reshape(out_bias, [W_out.shape[0], 1])

    return W, W_in, W_out, out_bias

def esn_sample_gen(W, W_in, W_out, out_bias, leaky_ratio, activation_fun, inputs):
    # simulate the ESN state space model
    num_units = W.shape[0]
    num_outputs = W_out.shape[0]
    num_inputs = W_in.shape[1]
    x_pre = np.zeros((num_units,1)) # initiate state as zeros if esn model use default zero initial state
    g_sample_all = np.zeros((num_units, inputs.shape[1])) # store all the activation function (function g() in paper) values, will be used as samples for MOR
    g_sample_stable_all = np.zeros((num_units, inputs.shape[1])) # store all the activation function (function g() in paper) values, will be used as samples for stable MOR
    x_sample_all = np.zeros((num_units, inputs.shape[1])) # store all the states, will be used as samples for training and MOR
    n_time = inputs[0].shape[0]
    inputs_reshape = tf.reshape(inputs[0,:,:], [n_time,num_inputs]).numpy().T
    g_sample = np.zeros((num_units,1))
    for i in range(inputs[0].shape[0]):
        if activation_fun == 'tanh':
            g_sample = leaky_ratio*np.tanh(W@x_pre + W_in@inputs_reshape[:,[i]])
            g_sample_stable = leaky_ratio*(np.tanh(W@x_pre + W_in@inputs_reshape[:,[i]])-W@x_pre)
        elif activation_fun == 'relu':
            g_sample = leaky_ratio*tf.nn.relu(W@x_pre + W_in@inputs_reshape[:,[i]])
            g_sample_stable = leaky_ratio*(np.relu(W@x_pre + W_in@inputs_reshape[:,[i]])-W@x_pre)
        else:
            raise Exception("activation function can only be tanh or relu")
        x_cur = (1-leaky_ratio)*x_pre + g_sample
        g_sample_all[:,[i]] = g_sample # record current activation function (function g() in paper) values as samples for MOR later
        g_sample_stable_all[:,[i]] = g_sample_stable # record current activation function (function g() in paper) values as samples for stable MOR later
        x_sample_all[:,[i]] = x_cur # record current state in all state vector as samples for training and MOR
        x_pre = x_cur
    return g_sample_all, g_sample_stable_all, x_sample_all

def esn_ss_sim(W, W_in, W_out, out_bias, leaky_ratio, activation_fun, inputs):
    # simulate the ESN state space model
    num_units = W.shape[0]
    num_outputs = W_out.shape[0]
    num_inputs = W_in.shape[1]
    x_pre = np.zeros((num_units,1)) # initiate state as zeros if esn model use default zero initial state
    # x_pre = np.zeros((num_units,1)) # initiate state as zeros if esn model use default zero initial state
    y_out = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
    n_time = inputs[0].shape[0]
    inputs_reshape = tf.reshape(inputs[0,:,:], [n_time,num_inputs]).numpy().T
    g_sample = np.zeros((num_units,1))
    for i in range(n_time):
        if activation_fun == 'tanh':
            # g_sample = leaky_ratio*np.tanh(W@x_pre + W_in@tf.reshape(inputs[0,i,:],[num_inputs,1]))
            g_sample = leaky_ratio*np.tanh(W@x_pre + W_in@inputs_reshape[:, [i]])
        elif activation_fun == 'relu':
            g_sample = leaky_ratio*tf.nn.relu(W@x_pre + W_in@inputs_reshape[:,[i]])
        else:
            raise Exception("activation function can only be tanh or relu")
        x_cur = (1-leaky_ratio)*x_pre + g_sample
        y_out[:,[i]] = W_out @ x_cur + out_bias
        x_pre = x_cur
    return y_out

def esn_ss_sim_sp(W_s, W_in, W_out, out_bias, leaky_ratio, activation_fun, inputs):
    # simulate the ESN state space model with sparse internal weight matrix (W_s)
    num_units = W_s.shape[0]
    num_outputs = W_out.shape[0]
    num_inputs = W_in.shape[1]
    x_pre = np.zeros((num_units,1)) # initiate state as zeros if esn model use default zero initial state
    # x_pre = np.zeros((num_units,1)) # initiate state as zeros if esn model use default zero initial state
    y_out = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
    n_time = inputs[0].shape[0]
    inputs_reshape = tf.reshape(inputs[0,:,:], [n_time,num_inputs]).numpy().T
    g_sample = np.zeros((num_units,1))
    for i in range(n_time):
        if activation_fun == 'tanh':
            # g_sample = leaky_ratio*np.tanh(W@x_pre + W_in@tf.reshape(inputs[0,i,:],[num_inputs,1]))
            g_sample = leaky_ratio*np.tanh(W_s*x_pre + W_in@inputs_reshape[:, [i]])
        elif activation_fun == 'relu':
            g_sample = leaky_ratio*tf.nn.relu(W_s*x_pre + W_in@inputs_reshape[:,[i]])
        else:
            raise Exception("activation function can only be tanh or relu")
        x_cur = (1-leaky_ratio)*x_pre + g_sample
        y_out[:,[i]] = W_out @ x_cur + out_bias
        x_pre = x_cur
    return y_out

def state_approx_sim(W_V_right, W_in, W_out_r, out_bias, V_left, leaky_ratio, activation_fun, inputs):
    # simulate the reduced ESN state space model without DEIM
    print("** simulating the reduced model...")
    order = W_V_right.shape[1]
    num_outputs = W_out_r.shape[0]
    num_inputs = W_in.shape[1]
    x_pre_r = np.zeros((order,1)) # initiate state as zeros if esn model use default zero initial state
    y_out_r = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
    x_cur_r = np.zeros((order,1))
    # x_sample_r_all = np.zeros((order, inputs.shape[1])) # store all the states, will be used as samples for training
    n_time = inputs[0].shape[0]
    inputs_reshape = tf.reshape(inputs[0,:,:], [n_time,num_inputs]).numpy().T
    for i in range(inputs[0].shape[0]):
        if activation_fun == 'tanh':
            x_cur_r = (1-leaky_ratio)*x_pre_r + leaky_ratio*V_left.T@np.tanh(W_V_right@x_pre_r + W_in@inputs_reshape[:,[i]])
        elif activation_fun == 'relu':
            x_cur_r = (1-leaky_ratio)*x_pre_r + leaky_ratio*V_left.T@tf.nn.relu(W_V_right@x_pre_r + W_in@inputs_reshape[:,[i]])
        else:
            raise Exception("activation function can only be tanh or relu")
        # x_sample_r_all[:,[i]] = x_cur_r # record current state in all state vector as samples for training
        y_out_r[:,[i]] = W_out_r @ x_cur_r + out_bias
        x_pre_r = x_cur_r
    # return y_out_r, x_sample_r_all
    return y_out_r

def esn_deim_sim(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, activation_fun, inputs):
    # simulate the reduced ESN model with DEIM
    print("** simulating the DEIM reduced model...")

    # t_unstable_before_start = time.process_time_ns()
    order = W_deim.shape[0]
    num_outputs = W_out_deim.shape[0]
    num_inputs = W_in_deim.shape[1]
    x_pre_deim = np.zeros((order,1)) # initiate state as zeros if esn model use default zero initial state
    y_out_deim = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
    x_cur_deim = np.zeros((order,1))
    x_sample_deim_all = np.zeros((order, inputs.shape[1])) # store all the states, will be used as samples for training
    # t_unstable_before = time.process_time_ns() - t_unstable_before_start
    # print("t_unstable_before: ", t_unstable_before)

    n_time = inputs[0].shape[0]
    inputs_reshape = tf.reshape(inputs[0,:,:], [n_time,num_inputs]).numpy().T

    for i in range(inputs[0].shape[0]):
        # t_unstable_loop_start = time.process_time_ns()
        if activation_fun == 'tanh':
            x_cur_deim = (1-leaky_ratio)*x_pre_deim + leaky_ratio*E_deim@np.tanh(W_deim@x_pre_deim + W_in_deim@inputs_reshape[:,[i]])
        elif activation_fun == 'relu':
            x_cur_deim = (1-leaky_ratio)*x_pre_deim + leaky_ratio*E_deim@tf.nn.relu(W_deim@x_pre_deim + W_in_deim@inputs_reshape[:,[i]])
        else:
            raise Exception("activation function can only be tanh or relu")
        # t_unstable_loop = time.process_time_ns() - t_unstable_loop_start
        # print("t_unstable_loop: ", t_unstable_loop)
        # if i < 4:
        #     print("first: ",W_deim@x_pre_deim)
        #     print("second: ",W_in_deim@tf.reshape(inputs[0,i,:],[num_inputs,1]))
        #     print("first+second: ",W_deim@x_pre_deim+W_in_deim@tf.reshape(inputs[0,i,:],[num_inputs,1]))
        #     print("tan(first+second): ",np.tanh(W_deim@x_pre_deim+W_in_deim@tf.reshape(inputs[0,i,:],[num_inputs,1])))
        #     print("Etan(first+second): ",E_deim@np.tanh(W_deim@x_pre_deim+W_in_deim@tf.reshape(inputs[0,i,:],[num_inputs,1])))
        # t_unstable_after_start = time.process_time_ns()
        x_sample_deim_all[:,[i]] = x_cur_deim # record current state in all state vector as samples for training
        y_out_deim[:,[i]] = W_out_deim @ x_cur_deim + out_bias
        x_pre_deim = x_cur_deim
        # t_unstable_after = time.process_time_ns() - t_unstable_after_start
        # print("t_unstable_after: ", t_unstable_after)
    return y_out_deim, x_sample_deim_all
    # return y_out_deim

def esn_deim_stable_sim(E_deim, E_lin, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, activation_fun, inputs):
    # simulate the reduced ESN model with DEIM
    print("** simulating the stable DEIM reduced model...")
    order = W_deim.shape[0]
    num_outputs = W_out_deim.shape[0]
    num_inputs = W_in_deim.shape[1]
    x_pre_deim = np.zeros((order,1)) # initiate state as zeros if esn model use default zero initial state
    x_cur_deim = np.zeros((order,1))
    y_out_deim = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
    # x_sample_deim_all = np.zeros((order, inputs.shape[1])) # store all the states, will be used as samples for training
    n_time = inputs[0].shape[0]
    inputs_reshape = tf.reshape(inputs[0,:,:], [n_time,num_inputs]).numpy().T

    for i in range(n_time):
        if activation_fun == 'tanh':
            # x_cur_deim = (1-leaky_ratio)*x_pre_deim + leaky_ratio*(E_lin@x_pre_deim + E_deim@np.tanh(W_deim@x_pre_deim + W_in_deim@tf.reshape(inputs[0,i,:],[num_inputs,1])))

            # x_cur_deim = W_in_deim@inputs_reshape[:,[i]]
            # x_cur_deim += W_deim@x_pre_deim
            # x_cur_deim = np.tanh(x_cur_deim)
            # x_cur_deim = E_deim@x_cur_deim
            # x_cur_deim += E_lin@x_pre_deim
            # x_cur_deim = leaky_ratio*x_cur_deim
            # x_cur_deim += (1-leaky_ratio)*x_pre_deim

            x_cur_deim = (1-leaky_ratio)*x_pre_deim + leaky_ratio*(E_lin@x_pre_deim + E_deim@np.tanh(W_deim@x_pre_deim + W_in_deim@inputs_reshape[:,[i]]))

        elif activation_fun == 'relu':
            x_cur_deim = (1-leaky_ratio)*x_pre_deim + leaky_ratio*(E_lin@x_pre_deim + E_deim@tf.nn.relu(W_deim@x_pre_deim + W_in_deim@inputs_reshape[:,[i]]))
        else:
            raise Exception("activation function can only be tanh or relu")
        # x_sample_deim_all[:,[i]] = x_cur_deim # record current state in all state vector as samples for training
        y_out_deim[:,[i]] = W_out_deim @ x_cur_deim + out_bias
        x_pre_deim = x_cur_deim
    # return y_out_deim, x_sample_deim_all
    return y_out_deim

def miniesn_unstable_assign(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, activation_fun, stime):
    # create reduced ESN network and assign weights
    print("** creating reduced ESN network without stablization and assigning weights...")
    order = W_deim.shape[0]
    num_inputs = W_in_deim.shape[1]
    num_outputs = W_out_deim.shape[0]
    recurrent_layer_miniesn_unstable =  miniESN_unstable(units=order, leaky=leaky_ratio, activation=activation_fun, connectivity=1, input_shape=(stime, num_inputs), return_sequences=True, use_bias=False, name="nn")
    output_miniesn_unstable = keras.layers.Dense(num_outputs, name="readouts")
    # put all together in a keras sequential model
    miniesn_unstable = keras.models.Sequential()
    miniesn_unstable.add(recurrent_layer_miniesn_unstable)
    miniesn_unstable.add(output_miniesn_unstable)
    miniesn_unstable.summary()
    miniesn_unstable.layers[0].weights[0].assign(tf.transpose(W_deim))
    miniesn_unstable.layers[0].weights[1].assign(tf.transpose(W_in_deim))
    miniesn_unstable.layers[0].weights[2].assign(tf.transpose(E_deim))
    W_out_deim = W_out_deim.astype('float32')
    miniesn_unstable.layers[1].weights[0].assign(tf.transpose(W_out_deim))
    out_bias = tf.reshape(out_bias, [num_outputs])
    miniesn_unstable.layers[1].weights[1].assign(tf.transpose(out_bias))
    return miniesn_unstable

def miniesn_stable_assign(E_deim_stable, W_deim_stable, W_in_deim_stable, W_out_deim_stable, E_lin_stable, out_bias, leaky_ratio, activation_fun, stime):
    # create reduced ESN network and assign weights
    print("** creating reduced ESN network with stablization and assigning weights...")
    order = W_deim_stable.shape[0]
    num_inputs = W_in_deim_stable.shape[1]
    num_outputs = W_out_deim_stable.shape[0]
    recurrent_layer_miniesn_stable =  miniESN_stable(units=order, leaky=leaky_ratio, activation=activation_fun, connectivity=1, input_shape=(stime, num_inputs), return_sequences=True, use_bias=False, name="nn")
    output_miniesn_stable = keras.layers.Dense(num_outputs, name="readouts")
    # put all together in a keras sequential model
    miniesn_stable = keras.models.Sequential()
    miniesn_stable.add(recurrent_layer_miniesn_stable)
    miniesn_stable.add(output_miniesn_stable)
    miniesn_stable.summary()
    miniesn_stable.layers[0].weights[0].assign(tf.transpose(W_deim_stable))
    miniesn_stable.layers[0].weights[1].assign(tf.transpose(W_in_deim_stable))
    miniesn_stable.layers[0].weights[2].assign(tf.transpose(E_deim_stable))
    miniesn_stable.layers[0].weights[3].assign(tf.transpose(E_lin_stable))
    W_out_deim_stable = W_out_deim_stable.astype('float32')
    miniesn_stable.layers[1].weights[0].assign(tf.transpose(W_out_deim_stable))
    out_bias = tf.reshape(out_bias, [num_outputs])
    miniesn_stable.layers[1].weights[1].assign(tf.transpose(out_bias))
    return miniesn_stable

def esn_train(x_all, y_train):
    # num_outputs = W_out.shape[0]
    # num_units = W_out.shape[1]
    # n_t = x_all.shape[1]
    # print("x_all_shape: ", x_all.shape)
    # print("y_train_shape: ", y_train.shape)
    # print("x_all: ", x_all)
    # print("x_all@x_all.T: ", x_all@(x_all.T))
    W_out = np.linalg.solve(x_all@(x_all.T),x_all@(y_train.T)).T
    # W_out = W_out.astype('float32') # convert to float to be compatible with tensorflow
    # W_out = np.linalg.solve(x_all.T,y_train.T).T
    return W_out

def esn_assign(model, W_out):
    print("** assigning the trained weights to ESN...")
    W_out = W_out.astype('float32')
    model.layers[1].weights[0].assign(tf.transpose(W_out))
    return model
