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
from scipy import sparse
import time

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

###################################### parameters ########################################
data_select = 4 # can only be 1, 2, 3, 4
stime_train = 100000 # sample number for training
stime_val = 10000 # sample number for validation
out_plt_index = 0 # the output to be plotted
in_plt_index = 0 # the input to be plotted
sample_step = 50 # the POD sample step (in time) in MOR, smaller value means finer sampling (more samples)
leaky_ratio = 1 # leaky ratio of ESN
connectivity_ratio = 0.1 # connectivity ratio of the ESN internal layer
activation_fun = 'tanh' # can only be 'tanh' or 'relu'
washout_end = 50 # the end point of the "washout" region in time series data
num_units_set = [500, 1000, 2000] # original ESN network hidden unit number
order_set = [10] # reduced order
num_test = 10 # number of test for each num_units-order combination

mset_esn_org = np.zeros((len(num_units_set), len(order_set)))
mset_ss_approx = np.zeros((len(num_units_set), len(order_set)))
mset_miniesn_unstable = np.zeros((len(num_units_set), len(order_set)))
mset_miniesn = np.zeros((len(num_units_set), len(order_set)))
mseo_ss_approx = np.zeros((len(num_units_set), len(order_set)))
mseo_miniesn_unstable = np.zeros((len(num_units_set), len(order_set)))
mseo_miniesn = np.zeros((len(num_units_set), len(order_set)))
runtime_esn_org = np.zeros((len(num_units_set), len(order_set)))
runtime_ss_approx = np.zeros((len(num_units_set), len(order_set)))
runtime_miniesn_unstable_ssim = np.zeros((len(num_units_set), len(order_set)))
# runtime_miniesn_unstable = np.zeros((len(num_units_set), len(order_set)))
runtime_miniesn = np.zeros((len(num_units_set), len(order_set)))

# loop for all num_units and order combinations; for each combination, perform test num_test times
for num_units_idx in range(0, len(num_units_set)):
    for order_idx in range(0, len(order_set)):

        mset_esn_org_tests = np.zeros(num_test)
        mset_ss_approx_tests = np.zeros(num_test)
        mset_miniesn_unstable_tests = np.zeros(num_test)
        mset_miniesn_tests = np.zeros(num_test)
        mseo_ss_approx_tests = np.zeros(num_test)
        mseo_miniesn_unstable_tests = np.zeros(num_test)
        mseo_miniesn_tests = np.zeros(num_test)
        runtime_esn_org_tests = np.zeros(num_test)
        runtime_ss_approx_tests = np.zeros(num_test)
        runtime_miniesn_unstable_ssim_tests = np.zeros(num_test)
        # runtime_miniesn_unstable_tests = np.zeros(num_test)
        runtime_miniesn_tests = np.zeros(num_test)
        
        for test_idx in range(0, num_test):
            
            num_units = num_units_set[num_units_idx]
            order = order_set[order_idx]
            
            ######################## generate data for training and validation ###############################
            if data_select == 1: # narma 10
                y_train, u_train = data_generate.narma_10_gen(stime_train)
                y_val, u_val = data_generate.narma_10_gen(stime_val)
            elif data_select == 2: # narma 30
                y_train, u_train = data_generate.narma_30_gen(stime_train)
                y_val, u_val = data_generate.narma_30_gen(stime_val)
            elif data_select == 3: # two-in-two-out dynamic system
                y_train, u_train = data_generate.two_in_two_out(stime_train)
                y_val, u_val = data_generate.two_in_two_out(stime_val)
            elif data_select == 4: # a second order problem
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

            # convert data to be compatible with tensorflow
            u_train = u_train.reshape(1,-1,num_inputs)
            y_train = y_train.reshape(1,-1,num_outputs)
            u_val = u_val.reshape(1,-1,num_inputs)
            y_val = y_val.reshape(1,-1,num_outputs)
            
            # plt.figure()
            # i, = plt.plot(u_train[0,:,out_plt_index], color="blue")
            # t, = plt.plot(y_train[0,:,out_plt_index], color="black", linestyle='dashed')
            # plt.xlabel("Timesteps")
            # plt.legend([i, t], ['input', "target"])

            ############### construct the original ESM (without training it yet) ######################
            recurrent_layer = tfa.layers.ESN(units=num_units, leaky=leaky_ratio, activation=activation_fun, connectivity=connectivity_ratio, input_shape=(stime_train, num_inputs), return_sequences=True, use_bias=False, name="nn")

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
            W_s = sparse.csr_matrix(W) # sparse W matrix
            
            # simulate the state space ESN model with training data to generate samples
            g_sample_all, g_sample_stable_all, x_sample_all = miniesn_tools.esn_sample_gen(W, W_in, W_out, out_bias, leaky_ratio, activation_fun, u_train)

            # train the original ESN
            W_out = miniesn_tools.esn_train(x_sample_all[:,washout_end:], y_train[0].T[:,washout_end:])
            # W_out = W_out.astype('float32')
            
            # assign the trained W_out back to ESN
            # model = miniesn_tools.esn_assign(model, W_out)

            # simulate the trained original ESN
            # t = time.process_time()
            t = time.perf_counter()
            # y_esn_val = model(u_val)
            # y_esn_val = miniesn_tools.esn_ss_sim(W, W_in, W_out, out_bias, leaky_ratio, activation_fun, u_val) # for dense W
            y_esn_val = miniesn_tools.esn_ss_sim_sp(W_s, W_in, W_out, out_bias, leaky_ratio, activation_fun, u_val) # for sparse W
            # runtime_esn_org_tests[test_idx] = time.process_time() - t
            runtime_esn_org_tests[test_idx] = time.perf_counter() - t

            ########## construct MiniESN using the trained ESN for simulation #######################

            # perform state approximation on ESN model
            W_out_r, V_left, V_right = miniesn_gen.state_approx(W, W_in, W_out, out_bias, x_sample_all[:,washout_end:], sample_step, order)

            # simulate the state approximate ESN without DEIM
            # t = time.process_time()
            W_V_right = W@V_right
            t = time.perf_counter()
            y_out_sa = miniesn_tools.state_approx_sim(W_V_right, W_in, W_out_r, out_bias, V_left, leaky_ratio, activation_fun, u_val)
            # runtime_ss_approx_tests[test_idx] = time.process_time() - t
            runtime_ss_approx_tests[test_idx] = time.perf_counter() - t
            
            # further perform DEIM to obtain miniESN without stabilization, this model is for demonstration ONLY
            W_deim, W_in_deim, E_deim, W_out_deim = miniesn_gen.miniesn_gen(W, W_in, W_out, V_left, V_right, g_sample_all[:,washout_end:], sample_step, order)

            # simulate miniESN with DEIM using state space model, for demonstration ONLY
            # t = time.process_time()
            t = time.perf_counter()
            y_out_deim, x_sample_deim = miniesn_tools.esn_deim_sim(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, activation_fun, u_val)
            # runtime_miniesn_unstable_ssim_tests[test_idx] = time.process_time() - t
            runtime_miniesn_unstable_ssim_tests[test_idx] = time.perf_counter() - t
            
            # # generate miniESN without stablization and assign weights, this network is for demonstration ONLY
            # miniesn_unstable = miniesn_tools.miniesn_unstable_assign(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, activation_fun, stime_val)

            # # simulate miniESN without stabilization
            # # t = time.process_time()
            # t = time.perf_counter()
            # y_out_miniesn_unstable = miniesn_unstable(u_val)
            # # runtime_miniesn_unstable_tests[test_idx] = time.process_time() - t
            # runtime_miniesn_unstable_tests[test_idx] = time.perf_counter() - t

            # perform stable DEIM to get the stable miniESN
            W_deim_stable, W_in_deim_stable, E_deim_stable, E_lin_stable, W_out_deim_stable = miniesn_gen.miniesn_stable(W, W_in, W_out, V_left, V_right, g_sample_stable_all[:,washout_end:], sample_step, order)

            # # generate stable miniESN and assign weights
            # miniesn_stable = miniesn_tools.miniesn_stable_assign(E_deim_stable, W_deim_stable, W_in_deim_stable, W_out_deim_stable, E_lin_stable, out_bias, leaky_ratio, activation_fun, stime_val)

            # simulate the stable miniESN
            # t = time.process_time()
            t = time.perf_counter()
            # y_out_miniesn_stable = miniesn_stable(u_val)
            y_out_miniesn_stable = miniesn_tools.esn_deim_stable_sim(E_deim_stable, E_lin_stable, W_deim_stable, W_in_deim_stable, W_out_deim_stable, out_bias, leaky_ratio, activation_fun, u_val)
            # runtime_miniesn_tests[test_idx] = time.process_time() - t
            runtime_miniesn_tests[test_idx] = time.perf_counter() - t
            
            ########################## compute the mse errors ############################
            mset_esn_org_tests[test_idx] = np.mean((y_val[0, washout_end:, :] - y_esn_val[:,washout_end:].T)**2)
            mset_ss_approx_tests[test_idx] = np.mean((y_val[0, washout_end:, :] - y_out_sa[:,washout_end:].T)**2)
            mset_miniesn_unstable_tests[test_idx] = np.mean((y_val[0, washout_end:, :] - y_out_deim[:,washout_end:].T)**2)
            mset_miniesn_tests[test_idx] = np.mean((y_val[0, washout_end:, :] - y_out_miniesn_stable[:,washout_end:].T)**2)
            mseo_ss_approx_tests[test_idx] = np.mean((y_esn_val[:,washout_end:].T - y_out_sa[:,washout_end:].T)**2)
            mseo_miniesn_unstable_tests[test_idx] = np.mean((y_esn_val[:,washout_end:].T - y_out_deim[:,washout_end:].T)**2)
            mseo_miniesn_tests[test_idx] = np.mean((y_esn_val[:,washout_end:].T - y_out_miniesn_stable[:,washout_end:].T)**2)

        # print("y_esn_val_type: ", y_esn_val.dtype)
        # print("y_out_sa_type: ", y_out_sa.dtype)
        # print("y_out_deim_type: ", y_out_deim.dtype)
        # print("y_out_miniesn_stable_type: ", y_out_miniesn_stable.dtype)
        # print(runtime_esn_org_tests)
        # print(runtime_ss_approx_tests)
        # print(runtime_esn_org_tests)
        # print(runtime_miniesn_unstable_ssim_tests)
        # print(runtime_miniesn_unstable_tests)
        # print(runtime_miniesn_tests)
        mset_esn_org[num_units_idx, order_idx] = np.mean(mset_esn_org_tests)
        mset_ss_approx[num_units_idx, order_idx] = np.mean(mset_ss_approx_tests)
        mset_miniesn_unstable[num_units_idx, order_idx] = np.mean(mset_miniesn_unstable_tests)
        mset_miniesn[num_units_idx, order_idx] = np.mean(mset_miniesn_tests)
        mseo_ss_approx[num_units_idx, order_idx] = np.mean(mseo_ss_approx_tests)
        mseo_miniesn_unstable[num_units_idx, order_idx] = np.mean(mseo_miniesn_unstable_tests)
        mseo_miniesn[num_units_idx, order_idx] = np.mean(mseo_miniesn_tests)
        runtime_esn_org[num_units_idx, order_idx] = np.mean(runtime_esn_org_tests)
        runtime_ss_approx[num_units_idx, order_idx] = np.mean(runtime_ss_approx_tests)
        runtime_miniesn_unstable_ssim[num_units_idx, order_idx] = np.mean(runtime_miniesn_unstable_ssim_tests)
        # runtime_miniesn_unstable[num_units_idx, order_idx] = np.mean(runtime_miniesn_unstable_tests)
        runtime_miniesn[num_units_idx, order_idx] = np.mean(runtime_miniesn_tests)

for num_units_idx in range(0, len(num_units_set)):
    for order_idx in range(0, len(order_set)):
        print("******************")
        print("num_units (original order): ", num_units_set[num_units_idx])
        print ("order: ", order_set[order_idx])
        print("mset_esn_org: ", mset_esn_org[num_units_idx, order_idx])
        print("mset_ss_approx: ", mset_ss_approx[num_units_idx, order_idx])
        print("mset_miniesn_unstable: ", mset_miniesn_unstable[num_units_idx, order_idx])
        print("mset_miniesn: ", mset_miniesn[num_units_idx, order_idx])
        print("mseo_ss_approx: ", mseo_ss_approx[num_units_idx, order_idx])
        print("mseo_miniesn_unstable: ", mseo_miniesn_unstable[num_units_idx, order_idx])
        print("mseo_miniesn: ", mseo_miniesn[num_units_idx, order_idx])
        print("runtime_esn_org: ", runtime_esn_org[num_units_idx, order_idx])
        print("runtime_ss_approx: ", runtime_ss_approx[num_units_idx, order_idx])
        print("speedup_ss_approx: ", runtime_esn_org[num_units_idx, order_idx]/runtime_ss_approx[num_units_idx, order_idx])
        print("runtime_miniesn_unstable_ssim: ", runtime_miniesn_unstable_ssim[num_units_idx, order_idx])
        print("speedup_miniesn_unstable_ssim: ", runtime_esn_org[num_units_idx, order_idx]/runtime_miniesn_unstable_ssim[num_units_idx, order_idx])
        # print("runtime_miniesn_unstable: ", runtime_miniesn_unstable[num_units_idx, order_idx])
        # print("speedup_miniesn_unstable: ", runtime_esn_org[num_units_idx, order_idx]/runtime_miniesn_unstable[num_units_idx, order_idx])
        print("runtime_miniesn: ", runtime_miniesn[num_units_idx, order_idx])
        print("speedup_miniesn: ", runtime_esn_org[num_units_idx, order_idx]/runtime_miniesn[num_units_idx, order_idx])



