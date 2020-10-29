import numpy as np
import data_generate
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow.keras as keras
import miniesn_gen
import miniesn_tools
import tensorflow as tf

###################################### parameters ########################################
data_select = 1 # can only be 1, 2, 3
stime_train = 5000 # sample number for training
stime_val = 200 # sample number for validation
epochs = 200
num_units = 500 # original ESN network hidden unit number
out_plt_index = 0 # the output to be plotted
in_plt_index = 0 # the input to be plotted
sample_step = 4 # the POD sample step (in time) in MOR, smaller value means finer sampling (more samples)
order = 20 # reduced order
leaky_ratio = 1 # leaky ratio of ESN
connectivity_ratio = 1 # connectivity ratio of the ESN internal layer
activation_fun = 'tanh' # can only be 'tanh' or 'relu'
washout_end = 50 # the end point of the "washout" region in time series data

######################## generate data for training and validation ###############################
if data_select == 1:
    y_train, u_train = data_generate.narma_10_gen(stime_train)
    y_val, u_val = data_generate.narma_10_gen(stime_val)
elif data_select == 2:
    y_train, u_train = data_generate.narma_30_gen(stime_train)
    y_val, u_val = data_generate.narma_30_gen(stime_val)
elif data_select == 3:
    y_train, u_train = data_generate.two_in_two_out(stime_train)
    y_val, u_val = data_generate.two_in_two_out(stime_val)
else:
    raise Exception("narma_order can only be 10 or 30 in this code")

num_inputs = u_train.shape[0]
num_outputs = y_train.shape[0]

u_train = u_train.T
y_train = y_train.T
u_val = u_val.T
y_val = y_val.T
# print("u_train: ", u_train)
# print("y_train: ", y_train)

u_train = u_train.reshape(1,-1,num_inputs)
y_train = y_train.reshape(1,-1,num_outputs)
u_val = u_val.reshape(1,-1,num_inputs)
y_val = y_val.reshape(1,-1,num_outputs)
plt.figure()
i, = plt.plot(u_train[0,:,out_plt_index], color="blue")
t, = plt.plot(y_train[0,:,out_plt_index], color="black", linestyle='dashed')
plt.xlabel("Timesteps")
plt.legend([i, t], ['input', "target"])

################ construct the original ESM (without training it yet) #########################
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

# extract the matrices before training
W, W_in, W_out, out_bias = miniesn_tools.esn_matrix_extract(model)

# simulate the untrained state space ESN model, y_untrained is inaccurate (of course since ESN is untrained), but g_sample_all and x_sample_all can still be used to train and reduce ESN
y_untrained, g_sample_all, g_sample_stable_all, x_sample_all = miniesn_tools.esn_ss_sim(W, W_in, W_out, out_bias, leaky_ratio, activation_fun, u_train)

########### construct the MiniESN without stabilization using the untrained original ESN model ##########

# perform MOR on the untrained ESN model
W_out_r, V_left, V_right = miniesn_gen.state_approx(W, W_in, W_out, out_bias, x_sample_all[:,washout_end:], sample_step, order)

# perform MOR with deim on the untrained ESN model
W_deim, W_in_deim, E_deim, W_out_deim = miniesn_gen.miniesn_gen(W, W_in, W_out, V_left, V_right, g_sample_all[:,washout_end:], sample_step, order)

# train miniESN using standard linear regression
# first, simulate MiniESN using training input to obtain the state samples for training (x_sample_deim_all_train), the output y_untrained_deim will not be used because it is inaccurate
y_untrained_deim, x_sample_deim_all_train = miniesn_tools.esn_deim_sim(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, activation_fun, u_train)
# training using linear regression
W_out_deim_lr = miniesn_tools.esn_train(x_sample_deim_all_train[:,washout_end:], y_train[0].T[:,washout_end:])

# generate miniESN and assign weights
model_red_lr = miniesn_tools.esn_deim_assign(E_deim, W_deim, W_in_deim, W_out_deim_lr, out_bias, leaky_ratio, activation_fun, stime_train)

y_out_esn_red_lr = model_red_lr(u_val)

############### construct the stable miniESN using the untrained original ESN model ###############

# perform stable DEIM to get the stable miniESN
W_deim_stable, W_in_deim_stable, E_deim_stable, E_lin_stable, W_out_deim_stable = miniesn_gen.miniesn_stable(W, W_in, W_out, V_left, V_right, g_sample_stable_all[:,washout_end:], sample_step, order)

# train the stable miniESN 
# simulate the stable miniESN using training input to obtain the state samples for training (x_sample_deim_stable_all_train), the output y_untrained_deim_stable will not be used because it is inaccurate
y_untrained_deim_stable, x_sample_deim_stable_all_train = miniesn_tools.esn_deim_stable_sim(E_deim_stable, E_lin_stable, W_deim_stable, W_in_deim_stable, W_out_deim_stable, out_bias, leaky_ratio, activation_fun, u_train)
# training using linear regression
W_out_deim_stable_lr = miniesn_tools.esn_train(x_sample_deim_stable_all_train[:,washout_end:], y_train[0].T[:,washout_end:])

y_out_miniesn_stable, x_sample_deim_stable_tmp = miniesn_tools.esn_deim_stable_sim(E_deim_stable, E_lin_stable, W_deim_stable, W_in_deim_stable, W_out_deim_stable_lr, out_bias, leaky_ratio, activation_fun, u_val)

###################################### train the original ESN ######################################
# linear regression based training
W_out = miniesn_tools.esn_train(x_sample_all[:,washout_end:], y_train[0].T[:,washout_end:])

# assign the trained W_out back to ESN
model = miniesn_tools.esn_assign(model, W_out)

# simulate the original ESN
y_esn_val = model(u_val)


############### construct an ESN with the same size of MiniESN, for accuracy comparison #################
recurrent_layer_small = tfa.layers.ESN(units=order, leaky=leaky_ratio, activation=activation_fun, connectivity=connectivity_ratio, input_shape=(stime_train, num_inputs), return_sequences=True, use_bias=False, name="nn")

# Build the readout layer
output_small = keras.layers.Dense(num_outputs, name="readouts")
# initialize the adam optimizer for training
optimizer = keras.optimizers.Adam(learning_rate=0.01)

# put all together in a keras sequential model
model_small = keras.models.Sequential()
model_small.add(recurrent_layer_small)
model_small.add(output_small)
model_small.summary()

# extract the matrices before training, W_p, W_in_p, out_bias_p should be the same as W, W_in, out_bias later because they cannot be trained
W_small, W_in_small, W_out_small, out_bias_small = miniesn_tools.esn_matrix_extract(model_small)

# simulate the untrained state space ESN model, y_out_p_small is inaccurate (of course since ESN is untrained), but x_sample_all_p_small can still be used to train ESN
y_untrained_small, g_sample_all_small, g_sample_stable_all_small, x_sample_all_small = miniesn_tools.esn_ss_sim(W_small, W_in_small, W_out_small, out_bias_small, leaky_ratio, activation_fun, u_train)

W_out_small = miniesn_tools.esn_train(x_sample_all_small[:,washout_end:], y_train[0].T[:,washout_end:])

model_small = miniesn_tools.esn_assign(model_small, W_out_small)

# training the original ESN
# model_small.compile(loss="mse", optimizer=optimizer)
# model_small.summary()
# hist_small = model_small.fit(u_train, y_train, epochs=epochs, verbose=0)

# plt.figure()
# loss_small, = plt.plot(hist_small.history['loss'])
# logloss_small, = plt.plot(np.log10(hist_small.history['loss']))
# plt.legend([loss_small, logloss_small], ["loss","log10(loss)"])

y_esn_small_val = model_small(u_val)

########################## compute the mse errors ############################
mse_esn_org = np.mean((y_val[0, washout_end:, :] - y_esn_val[0, washout_end:, :])**2)
mse_esn_small = np.mean((y_val[0, washout_end:, :] - y_esn_small_val[0, washout_end:, :])**2)
mse_miniesn = np.mean((y_val[0, washout_end:, :] - y_out_miniesn_stable[:,washout_end:].T)**2)
print("mse_esn_org: ", mse_esn_org)
print("mse_esn_small: ", mse_esn_small)
print("mse_miniesn: ", mse_miniesn)

######################### plot the accuracy comparison results ##################################

plt.figure()
t, = plt.plot(y_val[0,:,out_plt_index], color="black")
o, = plt.plot(y_esn_val[0,:,out_plt_index], color="blue", linestyle='dotted')
st, = plt.plot(y_out_miniesn_stable[out_plt_index, :], color="red", linestyle='dashdot')
s, = plt.plot(y_esn_small_val[0,:,out_plt_index], color="green", linestyle='dashed')
plt.xlabel("Timesteps")
plt.legend([t, o, st, s], ["target", "ESN org", "miniESN", "ESN small"])

plt.show()

