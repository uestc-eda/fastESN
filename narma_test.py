import numpy as np
import data_generate
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow.keras as keras
import mor_esn
import tensorflow as tf

narma_order = 10 # can only be 10 or 30
stime_train = 1000 # sample number for training
stime_val = 200 # sample number for validation
epochs = 100
num_units = 50
num_inputs = 1
num_outputs = 1

# generate data for training and validation
if narma_order == 10:
    y_train, u_train = data_generate.narma_10_gen(stime_train)
    y_val, u_val = data_generate.narma_10_gen(stime_val)
elif narma_order == 30:
    y_train, u_train = data_generate.narma_30_gen(stime_train)
    y_val, u_val = data_generate.narma_30_gen(stime_val)
else:
    raise Exception("narma_order can only be 10 or 30 in this code")
u_train = u_train.reshape(1,-1,1)
y_train = y_train.reshape(1,-1,1)
u_val = u_val.reshape(1,-1,1)
y_val = y_val.reshape(1,-1,1)

# plt.figure()
# i, = plt.plot(u_train[0], color="blue")
# t, = plt.plot(y_train[0], color="black")
# plt.xlabel("Timesteps")
# plt.legend([i, t], ['input', "target"])

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

print("** ploting the final results...")
plt.figure()
i, = plt.plot(u_train[0], color="blue")
t, = plt.plot(y_train[0], color="black")
o, = plt.plot(y_esn_train[0], color="red", linestyle='dashed')
plt.xlabel("Timesteps")
plt.legend([i, t, o], ['input', "target", "readout"])

y_esn_val = model(u_val)

# extract the ESN model in state space form
W, W_in, W_out, out_bias = mor_esn.esn_matrix_extract(model)

# simulate the state space ESN model
y_out, x_all = mor_esn.esn_ss_sim(W, W_in, W_out, out_bias, u_val)

# perform MOR on ESN model
sample_step = 3
order = 20
W_r, W_in_r, W_out_r, V = mor_esn.mor_esn(W, W_in, W_out, out_bias, x_all, sample_step, order)

# simulate the reduced ESN model without DEIM
y_out_r = mor_esn.esn_red_sim(W, W_in, W_out_r, out_bias, V, u_val)

# perform MOR with deim
W_deim, W_in_deim, E_deim, W_out_deim = mor_esn.deim_whole(W, W_in, W_out, V, x_all, order)

# simulate the reduced ESN model with DEIM
y_out_deim = mor_esn.esn_deim_sim(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, u_val)

# form a reduced ESN network
recurrent_layer_red = tfa.layers.ESN(units=order, leaky=1, activation='tanh', connectivity=1, input_shape=(stime_train, num_inputs), return_sequences=True, use_bias=False, name="nn")
deim_layer = keras.layers.Dense(order, name="deim")
output_red = keras.layers.Dense(num_outputs, name="readouts")
# put all together in a keras sequential model
model_red = keras.models.Sequential()
model_red.add(recurrent_layer_red)
model_red.add(deim_layer)
model_red.add(output_red)
model_red.layers[0].weights[0].assign(tf.transpose(W_deim))
model_red.layers[0].weights[1].assign(tf.transpose(W_in_deim))
model_red.layers[1].weights[0].assign(tf.transpose(E_deim))
model_red.layers[2].weights[0].assign(tf.transpose(W_out_deim))
model_red.layers[2].weights[1].assign(tf.transpose(out_bias))
model_red.summary()

y_out_esn_red = model_red(u_val) 


plt.figure()
# i, = plt.plot(u_val[0], color="blue")
t, = plt.plot(y_val[0], color="black")
o, = plt.plot(y_esn_val[0], color="red", linestyle='solid')
m, = plt.plot(y_out[0], color="green", linestyle='dashed')
r, = plt.plot(y_out_r[0], color="magenta", linestyle='dotted')
d, = plt.plot(y_out_deim[0], color="blue", linestyle='dashdot')
nr = plt.plot(y_out_esn_red[0], color="yellow", linestyle='dotted')
plt.xlabel("Timesteps")
plt.legend([t, o, m, r, d], ["target", "readout", "ss model", "reduced", "deim red", "esn red"])

plt.show()


