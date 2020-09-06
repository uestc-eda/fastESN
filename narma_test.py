import numpy as np
import data_generate
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow.keras as keras
import mor_esn
import tensorflow as tf

data_select = 1 # can only be 1, 2, 3
stime_train = 2000 # sample number for training
stime_val = 200 # sample number for validation
epochs = 200
num_units = 100 # original ESN network hidden unit number
out_plt_index = 0 # the output to be plotted
in_plt_index = 0 # the input to be plotted
sample_step = 3 # the POD sample step (in time) in MOR, smaller value means finer sampling (more samples)
order = 20 # reduced order

# generate data for training and validation
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

recurrent_layer = tfa.layers.ESN(units=num_units, leaky=1, activation='tanh', connectivity=0.7, input_shape=(stime_train, num_inputs), return_sequences=True, use_bias=False, name="nn")

# Build the readout layer
output = keras.layers.Dense(num_outputs, name="readouts")
# initialize the adam optimizer for training
optimizer = keras.optimizers.Adam(learning_rate=0.01)

# put all together in a keras sequential model
model = keras.models.Sequential()
model.add(recurrent_layer)
model.add(output)

model.compile(loss="mse", optimizer=optimizer)
model.summary()

# inner = model.get_layer("nn")(inputs) 
# outputs = model(inputs) 

hist = model.fit(u_train, y_train, epochs=epochs, verbose=0)

plt.figure()
loss, = plt.plot(hist.history['loss'])
logloss, = plt.plot(np.log10(hist.history['loss']))
plt.legend([loss, logloss], ["loss","log10(loss)"])

y_esn_train = model(u_train) 

# print("** ploting the final results...")
# plt.figure()
# i, = plt.plot(u_train[0,:,in_plt_index], color="blue")
# t, = plt.plot(y_train[0,:,out_plt_index], color="black")
# o, = plt.plot(y_esn_train[0,:,out_plt_index], color="red", linestyle='dashed')
# plt.xlabel("Timesteps")
# plt.legend([i, t, o], ['input', "target", "readout"])

y_esn_val = model(u_val)

# extract the ESN model in state space form
W, W_in, W_out, out_bias = mor_esn.esn_matrix_extract(model)

# simulate the state space ESN model
y_out, x_all = mor_esn.esn_ss_sim(W, W_in, W_out, out_bias, u_val)

# perform MOR on ESN model
W_r, W_in_r, W_out_r, V = mor_esn.mor_esn(W, W_in, W_out, out_bias, x_all, sample_step, order)

# simulate the reduced ESN model without DEIM
y_out_r = mor_esn.esn_red_sim(W, W_in, W_out_r, out_bias, V, u_val)

# perform MOR with deim
W_deim, W_in_deim, E_deim, W_out_deim = mor_esn.deim_whole(W, W_in, W_out, V, x_all, order)

# simulate the reduced ESN model with DEIM
y_out_deim = mor_esn.esn_deim_sim(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, u_val)

# generate the reduced ESN network and assign weights
model_red = mor_esn.esn_deim_assign(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, stime_val)

# simulate the reduced ESN network
y_out_esn_red = model_red(u_val) 


plt.figure()
# i, = plt.plot(u_val[0], color="blue")
t, = plt.plot(y_val[0,:,out_plt_index], color="black")
o, = plt.plot(y_esn_val[0,:,out_plt_index], color="red", linestyle='solid')
# m, = plt.plot(y_out[out_plt_index,:], color="green", linestyle='dashed')
r, = plt.plot(y_out_r[out_plt_index,:], color="magenta", linestyle='dotted')
d, = plt.plot(y_out_deim[out_plt_index,:], color="blue", linestyle='dashdot')
n, = plt.plot(y_out_esn_red[0,:,out_plt_index], color="green", linestyle='dashed')
plt.xlabel("Timesteps")
plt.legend([t, o, r, d, n], ["target", "readout", "reduced", "deim red", "esn red"])

plt.show()


