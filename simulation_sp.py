import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1'
    # NUMEXPR_NUM_THREADS = '1'
    # MKL_NUM_THREADS = '1'
)

import numpy as np
import data_generate
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow.keras as keras
import miniesn_gen
import miniesn_tools
import tensorflow as tf
import time
from scipy import sparse

###################################### parameters ########################################
data_select = 2  # can only be 1, 2, 3, 4
stime_train = 100000  # sample number for training
stime_val = 10000  # sample number for validation
num_units = 1000  # original ESN network hidden unit number
out_plt_index = 0  # the output to be plotted
in_plt_index = 0  # the input to be plotted
sample_step = 50  # the POD sample step (in time) in MOR, smaller value means finer sampling (more samples)
leaky_ratio = 1  # leaky ratio of ESN
activation_fun = 'tanh'  # can only be 'tanh' or 'relu'
washout_end = 50  # the end point of the "washout" region in time series data
connectivity_set = [0.01,0.05,0.10,0.20]  # connectivity ratios
order_set = [10,20,40,80]

######################## generate data for training and validation ###############################
if data_select == 1:  # narma 10
    y_train, u_train = data_generate.narma_10_gen(stime_train)
    y_val, u_val = data_generate.narma_10_gen(stime_val)
elif data_select == 2:  # narma 30
    y_train, u_train = data_generate.narma_30_gen(stime_train)
    y_val, u_val = data_generate.narma_30_gen(stime_val)
elif data_select == 3:  # two-in-two-out dynamic system
    y_train, u_train = data_generate.two_in_two_out(stime_train)
    y_val, u_val = data_generate.two_in_two_out(stime_val)
elif data_select == 4:  # a second order problem
    y_train, u_train = data_generate.second_order_problem(stime_train)
    y_val, u_val = data_generate.second_order_problem(stime_val)
else:
    raise Exception("data type can only be 1, 2, 3, 4 in this code")

num_inputs = u_train.shape[0]
num_outputs = y_train.shape[0]

u_train = u_train.T
y_train = y_train.T
u_val = u_val.T
y_val = y_val.T
# print("u_train: ", u_train)
# print("y_train: ", y_train)

u_train = u_train.reshape(1, -1, num_inputs)
y_train = y_train.reshape(1, -1, num_outputs)
u_val = u_val.reshape(1, -1, num_inputs)
y_val = y_val.reshape(1, -1, num_outputs)
# plt.figure()
# i, = plt.plot(u_train[0, :, out_plt_index], color="blue")
# t, = plt.plot(y_train[0, :, out_plt_index], color="black", linestyle='dashed')
# plt.xlabel("Timesteps")
# plt.legend([i, t], ['input', "target"])

time_t = np.zeros((2, len(connectivity_set)))
spar_param = np.zeros((1, len(connectivity_set)))

for i in range(len(connectivity_set)):
    ############### construct the original ESN (without training it yet) ######################
    connectivity_ratio = connectivity_set[i]

    recurrent_layer = tfa.layers.ESN(units=num_units, leaky=leaky_ratio, activation=activation_fun,
                                     connectivity=connectivity_ratio, input_shape=(stime_train, num_inputs),
                                     return_sequences=True, use_bias=False, name="nn")

    # Build the readout layer
    output = keras.layers.Dense(num_outputs, name="readouts")
    # initialize the adam optimizer for training
    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    # put all together in a keras sequential model
    model = keras.models.Sequential()
    model.add(recurrent_layer)
    model.add(output)
    model.summary()

    ########## train the original ESN model #######################

    # extract the ESN model in state space form
    W, W_in, W_out, out_bias = miniesn_tools.esn_matrix_extract(model)
    W = W.numpy()
    W_in = W_in.numpy()
    W_out = W_out.numpy()
    out_bias = out_bias.numpy()
    W = W.astype('float64')
    W_in = W_in.astype('float64')
    W_out = W_out.astype('float64')
    out_bias = out_bias.astype('float64')
    # sparse W matrix
    W_s = sparse.csr_matrix(W)

    # simulate the state space ESN model with training data to generate samples
    g_sample_all, g_sample_stable_all, x_sample_all = miniesn_tools.esn_sample_gen(W, W_in, W_out, out_bias,
                                                                                   leaky_ratio, activation_fun, u_train)

    # train the original ESN
    W_out = miniesn_tools.esn_train(x_sample_all[:, washout_end:], y_train[0].T[:, washout_end:])

    # # assign the trained W_out back to ESN
    # model = miniesn_tools.esn_assign(model, W_out)

    # time_start = time.time()
    # simulate the trained original ESN
    # y_esn_val = miniesn_tools.esn_ss_sim(W, W_in, W_out, out_bias, leaky_ratio, activation_fun, u_val)
    # y_esn_val = model(u_val)
    time1 = time.time()
    y_esn_val_s = miniesn_tools.esn_ss_sim_sp(W_s, W_in, W_out, out_bias, leaky_ratio, activation_fun, u_val)
    time2 = time.time()
    # time_t[0, i - 1] = time1 - time_start
    time_t[0, i] = time2 - time1
    spar_param[0,i] = np.count_nonzero(W)+num_units*2+out_bias.shape[0]
    mse_esn_org = np.mean((y_val[0, washout_end:, :] - y_esn_val_s[:,washout_end:].T)**2)
    print("connectivity_ratio: ",connectivity_ratio)
    print("mset_esn_org: ", mse_esn_org)
    print("sparse time: ", time_t[0,i], "\nsparse param #: ", spar_param[0,i])

    for order in order_set:
        print("\norder: ",order)
        # perform state approximation on ESN model
        W_out_r, V_left, V_right = miniesn_gen.state_approx(W, W_in, W_out, out_bias, x_sample_all[:,washout_end:], sample_step, order)

        # perform stable DEIM to get the stable miniESN
        W_deim_stable, W_in_deim_stable, E_deim_stable, E_lin_stable, W_out_deim_stable = miniesn_gen.miniesn_stable(W, W_in, W_out, V_left, V_right, g_sample_stable_all[:,washout_end:], sample_step, order)
        time3 = time.time()
        # simulate the stable miniESN using state space model
        y_out_miniesn_stable = miniesn_tools.esn_deim_stable_sim(E_deim_stable, E_lin_stable, W_deim_stable, W_in_deim_stable, W_out_deim_stable, out_bias, leaky_ratio, activation_fun, u_val)
        time4 = time.time()
        time_t[1, i] = time4 - time3
        mse_miniesn = np.mean((y_val[0, washout_end:, :] - y_out_miniesn_stable[:,washout_end:].T)**2)
        mseo_miniesn = np.mean((y_esn_val_s[:,washout_end:].T - y_out_miniesn_stable[:,washout_end:].T)**2)
        miniesn_parm=order*order*3+order*2+out_bias.shape[0]
        print("mset_miniesn: ", mse_miniesn, "\nmseo_miniesn: ", mseo_miniesn)
        print("miniesn time: ", time_t[1,i], "\nminiesn param #: ", miniesn_parm)
    print("\n")
