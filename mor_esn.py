import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from esn_red import ESNRed

def mor_esn(W, W_in, W_out, out_bias, sample_all, sample_step, order):
    # sample_step = x_all.shape[1]//n_sample # integer floor division "//" to compute integer sample_step
    # print("x_all=", x_all)
    print("sample_step=", sample_step)
    samples = sample_all[:, 0::sample_step]
    # print("samples=", samples)
    U, S, V = np.linalg.svd(samples, full_matrices=False)
    U = U[:,0:order]

    W_r = U.T@W@U
    W_in_r = U.T@W_in
    W_out_r = W_out@U

    return W_r, W_in_r, W_out_r, U

def deim_whole(W, W_in, W_out, V, sample_all, sample_step, order_deim):
    samples = sample_all[:, 0::sample_step]
    U, S, V_deim = np.linalg.svd(samples, full_matrices=False)
    U = U[:,0:order_deim]

    idx, P = deim_core(U)

    W_deim = P.T@W@V
    W_in_deim = P.T@W_in
    E_deim = V.T@U@np.linalg.inv(P.T@U)
    E_deim = E_deim.astype('float32') # convert to float to be compatible with tensorflow
    W_out_deim = W_out@V

    return W_deim, W_in_deim, E_deim, W_out_deim

def deim_core(U):
    n = U.shape[0] # original model order
    order_deim = U.shape[1] # reduced model deim order
    # U_out = np.zeros([n, order_deim])

    P = np.zeros((n, order_deim)) # initiate P matrix
    idx = np.zeros(order_deim, dtype=int) # initial index vector

    idx[0] = np.argmax(np.absolute(U[:,0]))
    # U_out[:,0] = U[:,0]
    # print("U_out=", U_out)
    P[idx[0],0] = 1 # set the first column of P

    # tmp = P[:,0:1].T@U[:,0:1]
    # print("tmp=", tmp)
    
    for i in range(1,order_deim): # loop start from the second index
        c = np.linalg.solve(P[:,0:i].T@U[:,0:i], P[:,0:i].T@U[:,i])
        # print("c=",c)
        res = U[:,i] - U[:,0:i]@c
        idx[i] = np.argmax(np.absolute(res))
        P[idx[i],i] = 1 # set the ith column of P

    # print("P=", P)
    print("idx=", idx)
    return idx, P

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
    sample_all = np.zeros((num_units, inputs.shape[1])) # store all the states, will be used as samples for MOR
    y_out = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
    for i in range(inputs[0].shape[0]):    
        sample = leaky_ratio*np.tanh(W@x_pre + W_in@tf.reshape(inputs[0,i,:],[num_outputs,1]))
        x_cur = (1-leaky_ratio)*x_pre + sample
        y_out[:,[i]] = W_out @ x_cur + out_bias
        sample_all[:,[i]] = sample # record current state in all state vector as samples for MOR later
        x_pre = x_cur
    return y_out, sample_all

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
    for i in range(inputs[0].shape[0]):    
        x_cur_deim = (1-leaky_ratio)*x_pre_deim + leaky_ratio*E_deim@np.tanh(W_deim@x_pre_deim + W_in_deim@tf.reshape(inputs[0,i,:],[num_outputs,1]))
        y_out_deim[:,[i]] = W_out_deim @ x_cur_deim + out_bias
        x_pre_deim = x_cur_deim
    return y_out_deim

def esn_deim_assign(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, stime):
    # create reduced ESN network and assign weights
    print("** creating reduced ESN network and assigning weights...")
    order = W_deim.shape[0]
    num_inputs = W_in_deim.shape[1]
    num_outputs = W_out_deim.shape[0]
    recurrent_layer_red =  ESNRed(units=order, leaky=leaky_ratio, activation='tanh', connectivity=1, input_shape=(stime, num_inputs), return_sequences=True, use_bias=False, name="nn")
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
