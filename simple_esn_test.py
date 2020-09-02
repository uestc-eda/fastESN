import sys, os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.ops import math_ops
from ESN import EchoStateRNNCell
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import mor_esn

# memory growth
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# random numbers
random_seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
print("seed: ", random_seed)

# batches = 100
stime = 100
epochs = 1000
num_units = 100
num_inputs = 1
num_outputs = 1
   
activation = lambda x: math_ops.tanh(x)

t = np.linspace(0, 1, stime).reshape(1,-1, 1).astype("float32")
# inputs = np.exp(-t*200)
inputs = np.exp(np.sin(t*6*np.pi))**2
targets = np.cos(t*6*np.pi)
# targets = np.cos(t*6*np.pi)*(t**(-0.1))

plt.figure()
i, = plt.plot(inputs[0], color="blue")
t, = plt.plot(targets[0], color="black")
plt.xlabel("Timesteps")
plt.legend([i, t], ['input', "target"])

# # Init the ESN cell
# cell = EchoStateRNNCell(units=num_units, 
#                         activation=activation, 
#                         decay=0.1, 
#                         epsilon=1e-20,
#                         alpha=0.5,
#                         optimize=True,
#                         optimize_vars=["rho", "decay", "alpha", "sw"],
#                         seed=random_seed)

# # Build the recurrent layer containing the ESN cell
# recurrent_layer = keras.layers.RNN(cell, input_shape=(stime, num_inputs), 
#                                    return_sequences=True, name="nn")

recurrent_layer = tfa.layers.ESN(units=num_units, leaky=1, activation='tanh', connectivity=0.1, input_shape=(stime, num_inputs), return_sequences=True, use_bias=False, name="nn")

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

inner = model.get_layer("nn")(inputs) 
outputs = model(inputs) 

plt.figure()
r = plt.plot(inner.numpy()[0], color = "black", alpha=0.3)
o, = plt.plot(outputs.numpy()[0], lw=5, color ="#880000")
plt.xlabel("Timesteps")
plt.ylim([-1.1,1.5])
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.legend([r[0],o], ["network activity", "readout"])

hist = model.fit(inputs, targets, epochs=epochs, verbose=0)

plt.figure()
loss, = plt.plot(hist.history['loss'])
logloss, = plt.plot(np.log10(hist.history['loss']))
plt.legend([loss, logloss], ["loss","log10(loss)"])

inner = model.get_layer("nn")(inputs) 
outputs = model(inputs) 

plt.figure()
r = plt.plot(inner.numpy()[0], color = "black", alpha=0.3)
o, = plt.plot(outputs.numpy()[0], lw=5, color ="#880000")
plt.xlabel("Timesteps")
plt.ylim([-1.1,1.5])
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.legend([r[0],o], ["network activity", "readout"])

# plt.show()
# print(model.layers[0].weights)
# print(model.layers[1].weights)

# extract the ESN model in state space form
W = tf.transpose(model.layers[0].weights[0])
W_in = tf.transpose(model.layers[0].weights[1])
# print("W_in=", W_in)
# print("W=", W)

W_out = tf.transpose(model.layers[1].weights[0])
out_bias = tf.transpose(model.layers[1].weights[1])
# print("W_out=", W_out)
# print("out_bias=", out_bias)

# simulate the state space ESN model
x_pre = np.zeros((num_units,1)) # initiate state as zeros if esn model use default zero initial state
x_all = np.zeros((num_units, inputs.shape[1])) # store all the states, will be used as samples for MOR
y_out = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
# print("shape_inputs: ", inputs.shape)
for i in range(inputs[0].shape[0]):    
    x_cur = np.tanh(W@x_pre + W_in@tf.reshape(inputs[0,i,:],[1,1]))
    y_out[:,i] = W_out @ x_cur + out_bias
    x_all[:,[i]] = x_cur # record current state in all state vector as samples for MOR later
    x_pre = x_cur
# print("y_out=", y_out)

# perform MOR on ESN model
sample_step = 3
order = 20
W_r, W_in_r, W_out_r, V = mor_esn.mor_esn(W, W_in, W_out, out_bias, x_all, sample_step, order)
# print("W_r=", W_r)
# print("W_in_r=", W_in_r)
# print("W_out_r=", W_out_r)
# print("V=", V)

# simulate the reduced ESN model without DEIM
print("** simulating the reduced model...")
x_pre_r = np.zeros((order,1)) # initiate state as zeros if esn model use default zero initial state
y_out_r = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
# print("shape_inputs: ", inputs.shape)
for i in range(inputs[0].shape[0]):    
    x_cur_r = V.T@np.tanh(W@V@x_pre_r + W_in@tf.reshape(inputs[0,i,:],[1,1]))
    y_out_r[:,i] = W_out_r @ x_cur_r + out_bias
    x_pre_r = x_cur_r
# print("y_out_r=", y_out_r)

W_deim, W_in_deim, E_deim, W_out_deim = mor_esn.deim_whole(W, W_in, W_out, V, x_all, order)

# simulate the reduced ESN model with DEIM
print("** simulating the DEIM reduced model...")
x_pre_deim = np.zeros((order,1)) # initiate state as zeros if esn model use default zero initial state
y_out_deim = np.zeros((num_outputs, inputs.shape[1])) # output matrix, composed of output vectors over time
for i in range(inputs[0].shape[0]):    
    x_cur_deim = E_deim@np.tanh(W_deim@x_pre_deim + W_in_deim@tf.reshape(inputs[0,i,:],[1,1]))
    y_out_deim[:,i] = W_out_deim @ x_cur_deim + out_bias
    x_pre_deim = x_cur_deim

# plot the results
print("** ploting the final results...")
plt.figure()
outputs = model(inputs) 
i, = plt.plot(inputs[0], color="cyan")
t, = plt.plot(targets[0], color="#44ff44", lw=10)
o, = plt.plot(outputs[0], color="#ff4444", lw=4)
m, = plt.plot(y_out[0], color="black", linestyle='dashed')
r, = plt.plot(y_out_r[0], color="magenta", linestyle='dotted')
d, = plt.plot(y_out_deim[0], color="blue", linestyle='dashdot')
plt.xlabel("Timesteps")
plt.legend([i, t, o, m, r, d], ['input', "target", "readout", "ss model", "reduced", "deim red"])

plt.show()
