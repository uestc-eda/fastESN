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
stime_train = 2000 # sample number for training
stime_val = 200 # sample number for validation
epochs = 200
num_units = 200 # original ESN network hidden unit number
out_plt_index = 0 # the output to be plotted
in_plt_index = 0 # the input to be plotted
sample_step = 3 # the POD sample step (in time) in MOR, smaller value means finer sampling (more samples)
order = 20 # reduced order
leaky_ratio = 0.7 # leaky ratio of ESN

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
recurrent_layer = tfa.layers.ESN(units=num_units, leaky=leaky_ratio, activation='tanh', connectivity=0.7, input_shape=(stime_train, num_inputs), return_sequences=True, use_bias=False, name="nn")

# Build the readout layer
output = keras.layers.Dense(num_outputs, name="readouts")
# initialize the adam optimizer for training
optimizer = keras.optimizers.Adam(learning_rate=0.01)

# put all together in a keras sequential model
model = keras.models.Sequential()
model.add(recurrent_layer)
model.add(output)

################## construct the MiniESN using the untrained original ESN model, then train the MiniESN #######################

# extract the matrices before training, W_p, W_in_p, out_bias_p should be the same as W, W_in, out_bias later because they cannot be trained
W_p, W_in_p, W_out_p, out_bias_p = miniesn_tools.esn_matrix_extract(model)

# simulate the untrained state space ESN model, y_out_p is inaccurate (of course since ESN is untrained), but sample_all_p can still be used to reduce ESN
y_out_p, sample_all_p = miniesn_tools.esn_ss_sim(W_p, W_in_p, W_out_p, out_bias_p, leaky_ratio, u_val)

# perform MOR on the untrained ESN model
W_r_p, W_in_r_p, W_out_r_p, V_p = miniesn_gen.mor_esn(W_p, W_in_p, W_out_p, out_bias_p, sample_all_p, sample_step, order)

# perform MOR with deim on the untrained ESN model
W_deim_p, W_in_deim_p, E_deim_p, W_out_deim_p = miniesn_gen.miniesn_gen(W_p, W_in_p, W_out_p, V_p, sample_all_p, sample_step, order)

# generate the untrained reduced ESN network and assign weights
model_red_p = miniesn_tools.esn_deim_assign(E_deim_p, W_deim_p, W_in_deim_p, W_out_deim_p, out_bias_p, leaky_ratio, stime_train)

# training the reduced ESN
model_red_p.compile(loss="mse", optimizer=optimizer)
hist_p = model_red_p.fit(u_train, y_train, epochs=epochs, verbose=0)

plt.figure()
loss_p, = plt.plot(hist_p.history['loss'])
logloss_p, = plt.plot(np.log10(hist_p.history['loss']))
plt.legend([loss_p, logloss_p], ["loss","log10(loss)"])

y_out_esn_red_p = model_red_p(u_val)

###################################### train the original ESN ######################################
# training the original ESN
model.compile(loss="mse", optimizer=optimizer)
model.summary()
hist = model.fit(u_train, y_train, epochs=epochs, verbose=0)

plt.figure()
loss, = plt.plot(hist.history['loss'])
logloss, = plt.plot(np.log10(hist.history['loss']))
plt.legend([loss, logloss], ["loss","log10(loss)"])

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

# simulate the state space ESN model
y_out, sample_all = miniesn_tools.esn_ss_sim(W, W_in, W_out, out_bias, leaky_ratio, u_val)

# perform MOR on ESN model
W_r, W_in_r, W_out_r, V = miniesn_gen.mor_esn(W, W_in, W_out, out_bias, sample_all, sample_step, order)

# simulate the reduced ESN model without DEIM
y_out_r = miniesn_tools.esn_red_sim(W, W_in, W_out_r, out_bias, V, leaky_ratio, u_val)

# perform MOR with deim
W_deim, W_in_deim, E_deim, W_out_deim = miniesn_gen.miniesn_gen(W, W_in, W_out, V, sample_all, sample_step, order)

# simulate the reduced ESN model with DEIM
y_out_deim = miniesn_tools.esn_deim_sim(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, u_val)

# generate the reduced ESN network and assign weights
model_red = miniesn_tools.esn_deim_assign(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, stime_val)

# simulate the reduced ESN network
y_out_esn_red = model_red(u_val) 

############### construct an ESN with the same size of MiniESN, for accuracy comparison #################
recurrent_layer_small = tfa.layers.ESN(units=order, leaky=leaky_ratio, activation='tanh', connectivity=1, input_shape=(stime_train, num_inputs), return_sequences=True, use_bias=False, name="nn")

# Build the readout layer
output_small = keras.layers.Dense(num_outputs, name="readouts")
# initialize the adam optimizer for training
optimizer = keras.optimizers.Adam(learning_rate=0.01)

# put all together in a keras sequential model
model_small = keras.models.Sequential()
model_small.add(recurrent_layer_small)
model_small.add(output_small)

# training the original ESN
model_small.compile(loss="mse", optimizer=optimizer)
model_small.summary()
hist_small = model_small.fit(u_train, y_train, epochs=epochs, verbose=0)

plt.figure()
loss_small, = plt.plot(hist_small.history['loss'])
logloss_small, = plt.plot(np.log10(hist_small.history['loss']))
plt.legend([loss_small, logloss_small], ["loss","log10(loss)"])

y_esn_small_val = model_small(u_val)

######################### plot the accuracy comparison results ##################################

plt.figure()
# i, = plt.plot(u_val[0], color="blue")
t, = plt.plot(y_val[0,:,out_plt_index], color="black")
o, = plt.plot(y_esn_val[0,:,out_plt_index], color="red", linestyle='solid')
# m, = plt.plot(y_out[out_plt_index,:], color="yellow", linestyle='dashed')
r, = plt.plot(y_out_r[out_plt_index,:], color="magenta", linestyle='dotted')
# d, = plt.plot(y_out_deim[out_plt_index,:], color="blue", linestyle='dashdot')
n, = plt.plot(y_out_esn_red[0,:,out_plt_index], color="green", linestyle='dashed')
e, = plt.plot(y_out_esn_red_p[0,:,out_plt_index], color="blue", linestyle='dashdot')
s, = plt.plot(y_esn_small_val[0,:,out_plt_index], color="yellow", linestyle='dashed')
plt.xlabel("Timesteps")
plt.legend([t, o, r, n, e, s], ["target", "ESN org", "reduced", "MiniESN with trained ESN", "MiniESN self trained", "ESN small"])

plt.show()


