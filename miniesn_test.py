import numpy as np
import data_generate
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow.keras as keras
import miniesn_gen
import miniesn_tools
import tensorflow as tf

###################################### parameters ########################################
data_select = 3 # can only be 1, 2, 3
stime_train = 1000 # sample number for training
stime_val = 200 # sample number for validation
epochs = 200
num_units = 100 # original ESN network hidden unit number
out_plt_index = 0 # the output to be plotted
in_plt_index = 0 # the input to be plotted
sample_step = 4 # the POD sample step (in time) in MOR, smaller value means finer sampling (more samples)
order = 60 # reduced order
leaky_ratio = 0.5 # leaky ratio of ESN
connectivity_ratio = 1 # connectivity ratio of the ESN internal layer
activation_fun = 'tanh' # can only be 'tanh' or 'relu'

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

############################ construct the original ESM (without training it yet) #######################################
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

################## construct the MiniESN using the untrained original ESN model, then train the MiniESN #######################

# extract the matrices before training, W_p, W_in_p, out_bias_p should be the same as W, W_in, out_bias later because they cannot be trained
W_p, W_in_p, W_out_p, out_bias_p = miniesn_tools.esn_matrix_extract(model)

# simulate the untrained state space ESN model, y_untrained_p is inaccurate (of course since ESN is untrained), but g_sample_all_p and x_sample_all_p can still be used to train and reduce ESN
y_untrained_p, g_sample_all_p, g_sample_stable_all_p, x_sample_all_p = miniesn_tools.esn_ss_sim(W_p, W_in_p, W_out_p, out_bias_p, leaky_ratio, activation_fun, u_train)

# perform MOR on the untrained ESN model
W_out_r_p, V_p = miniesn_gen.mor_esn(W_p, W_in_p, W_out_p, out_bias_p, x_sample_all_p, sample_step, order)

# perform MOR with deim on the untrained ESN model
W_deim_p, W_in_deim_p, E_deim_p, W_out_deim_p = miniesn_gen.miniesn_gen(W_p, W_in_p, W_out_p, V_p, g_sample_all_p, sample_step, order)

# train MiniESN using standard regression
# first, simulate MiniESN using training input to obtain the state samples for training (x_sample_deim_all_train), the output y_untrained_deim will not be used because it is inaccurate
y_untrained_deim, x_sample_deim_all_train = miniesn_tools.esn_deim_sim(E_deim_p, W_deim_p, W_in_deim_p, W_out_deim_p, out_bias_p, leaky_ratio, activation_fun, u_train)
# training using linear regression
W_out_deim_p_lr = miniesn_tools.esn_train(x_sample_deim_all_train, y_train[0].T)

# generate MiniESN and assign weights
model_red_p_lr = miniesn_tools.esn_deim_assign(E_deim_p, W_deim_p, W_in_deim_p, W_out_deim_p_lr, out_bias_p, leaky_ratio, activation_fun, stime_train)

y_out_esn_red_p_lr = model_red_p_lr(u_val)

# train MiniESN using back propagation
model_red_p_bp = model_red_p_lr # intiate network using model_red_p_lr
model_red_p_bp.compile(loss="mse", optimizer=optimizer)
hist_p_bp = model_red_p_bp.fit(u_train, y_train, epochs=epochs, verbose=0)

plt.figure()
loss_p_bp, = plt.plot(hist_p_bp.history['loss'])
logloss_p_bp, = plt.plot(np.log10(hist_p_bp.history['loss']))
plt.legend([loss_p_bp, logloss_p_bp], ["loss","log10(loss)"])

y_out_esn_red_p_bp = model_red_p_bp(u_val)

###################################### train the original ESN ######################################
# training the original ESN
# model.compile(loss="mse", optimizer=optimizer)
# model.summary()
# hist = model.fit(u_train, y_train, epochs=epochs, verbose=0)

# plt.figure()
# loss, = plt.plot(hist.history['loss'])
# logloss, = plt.plot(np.log10(hist.history['loss']))
# plt.legend([loss, logloss], ["loss","log10(loss)"])
W_out = miniesn_tools.esn_train(x_sample_all_p, y_train[0].T)

# simulate the state space ESN model
# y_out_test, g_sample_test, x_sample_test = miniesn_tools.esn_ss_sim(W_p, W_in_p, W_out, out_bias_p, leaky_ratio, u_train)

# y_out_test = W_out@sample_all_p
# plt.figure()
# t, = plt.plot(y_train[0,:,out_plt_index], color="black", linestyle='dotted')
# i, = plt.plot(y_out_test[out_plt_index,:], color="blue", linestyle='dashed')
# plt.xlabel("Timesteps")
# plt.legend([i, t], ['out_test', "target"])


# assign the trained W_out back to ESN
# model.layers[1].weights[0].assign(tf.transpose(W_out))
model = miniesn_tools.esn_assign(model, W_out)


# y_esn_train = model(u_train) 

# print("** ploting the final results...")
# plt.figure()
# i, = plt.plot(u_train[0,:,in_plt_index], color="blue")
# t, = plt.plot(y_train[0,:,out_plt_index], color="black")
# o, = plt.plot(y_esn_train[0,:,out_plt_index], color="red", linestyle='dashed')
# plt.xlabel("Timesteps")
# plt.legend([i, t, o], ['input', "target", "readout"])

y_esn_val = model(u_val)


########################### construct MiniESN using the trained ESN ###################################

# extract the ESN model in state space form
W, W_in, W_out, out_bias = miniesn_tools.esn_matrix_extract(model)

# simulate the state space ESN model with training data to generate samples
y_out_train, g_sample_all, g_sample_stable_all, x_sample_all = miniesn_tools.esn_ss_sim(W, W_in, W_out, out_bias, leaky_ratio, activation_fun, u_train)

# perform MOR on ESN model
W_out_r, V = miniesn_gen.mor_esn(W, W_in, W_out, out_bias, x_sample_all, sample_step, order)

# simulate the reduced ESN model without DEIM
y_out_r, x_sample_r = miniesn_tools.esn_red_sim(W, W_in, W_out_r, out_bias, V, leaky_ratio, activation_fun, u_val)

# perform MOR with deim
W_deim, W_in_deim, E_deim, W_out_deim = miniesn_gen.miniesn_gen(W, W_in, W_out, V, g_sample_all, sample_step, order)

# simulate the reduced ESN model with DEIM
y_out_deim, x_sample_deim = miniesn_tools.esn_deim_sim(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, activation_fun, u_val)

# generate the reduced ESN network and assign weights
model_red = miniesn_tools.esn_deim_assign(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, activation_fun, stime_val)

# simulate the reduced ESN network
y_out_esn_red = model_red(u_val)

# perform MOR with stable deim
W_deim_stable, W_in_deim_stable, E_deim_stable, E_lin_stable, W_out_deim_stable = miniesn_gen.miniesn_stable(W, W_in, W_out, V, g_sample_stable_all, sample_step, order)

# simulate the reduced ESN model with stable DEIM
y_out_deim_stable, x_sample_deim_stable = miniesn_tools.esn_deim_stable_sim(E_deim_stable, E_lin_stable, W_deim_stable, W_in_deim_stable, W_out_deim_stable, out_bias, leaky_ratio, activation_fun, u_val)

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

W_out_small = miniesn_tools.esn_train(x_sample_all_small, y_train[0].T)

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
mse_esn_small = np.mean((y_val[0, 50:, :] - y_esn_small_val[0, 50:, :])**2)
mse_miniesn_bp = np.mean((y_val[0, 50:, :] - y_out_esn_red_p_bp[0, 50:, :])**2)
mse_miniesn_lr = np.mean((y_val[0, 50:, :] - y_out_esn_red_p_lr[0, 50:, :])**2)
mse_esn_org = np.mean((y_val[0, 50:, :] - y_esn_val[0, 50:, :])**2)
print("mse_esn_small: ", mse_esn_small)
print("mse_miniesn_bp: ", mse_miniesn_bp)
print("mse_miniesn_lr: ", mse_miniesn_lr)
print("mse_esn_org: ", mse_esn_org)

######################### plot the accuracy comparison results ##################################

plt.figure()
# i, = plt.plot(u_val[0], color="blue")
t, = plt.plot(y_val[0,:,out_plt_index], color="black")
o, = plt.plot(y_esn_val[0,:,out_plt_index], color="red", linestyle='solid')
# m, = plt.plot(y_out[out_plt_index,:], color="yellow", linestyle='dashed')
# r, = plt.plot(y_out_r[out_plt_index,:], color="magenta", linestyle='dotted')
# n, = plt.plot(y_out_esn_red[0,:,out_plt_index], color="green", linestyle='dashed')
e_lr, = plt.plot(y_out_esn_red_p_lr[0,:,out_plt_index], color="blue", linestyle='dashdot')
e_bp, = plt.plot(y_out_esn_red_p_bp[0,:,out_plt_index], color="magenta", linestyle='dotted')
s, = plt.plot(y_esn_small_val[0,:,out_plt_index], color="green", linestyle='dashed')
plt.xlabel("Timesteps")
plt.legend([t, o, e_lr, e_bp, s], ["target", "ESN org", "MiniESN with lr", "MiniESN with bp", "ESN small"])

plt.figure()
t, = plt.plot(y_val[0,:,out_plt_index], color="black")
o, = plt.plot(y_esn_val[0,:,out_plt_index], color="red", linestyle='solid')
d, = plt.plot(y_out_esn_red[0,:,out_plt_index], color="green", linestyle='dashed')
r, = plt.plot(y_out_r[out_plt_index, :], color="blue", linestyle='dotted')
st, = plt.plot(y_out_deim_stable[out_plt_index, :], color="magenta", linestyle='dashdot')
plt.xlabel("Timesteps")
plt.legend([t, o, r, d, st], ["target", "ESN org", "SS Approx", "Red from ESN", "stable"])

plt.figure()
state, = plt.plot(x_sample_deim[1,:], color="black")
plt.xlabel("Timesteps")
plt.legend([state], ["state1"])

plt.show()


