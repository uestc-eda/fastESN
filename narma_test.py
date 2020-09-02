import numpy as np
import data_generate
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow.keras as keras

stime_train = 1000 # sample number for training
stime_val = 200 # sample number for validation
epochs = 1000
num_units = 200
num_inputs = 1
num_outputs = 1

# generate data for training and validation
# y_30, u_30 = data_generate.narma_30_gen(n_t)
y_10_train, u_10_train = data_generate.narma_10_gen(stime_train)
y_10_val, u_10_val = data_generate.narma_10_gen(stime_val)
u_10_train = u_10_train.reshape(1,-1,1)
y_10_train = y_10_train.reshape(1,-1,1)
u_10_val = u_10_val.reshape(1,-1,1)
y_10_val = y_10_val.reshape(1,-1,1)
# t_train = np.linspace(0, 1, stime_train)
# t_val = np.linspace(0, 1, stime_val)

# plt.figure()
# i, = plt.plot(u_10_train[0], color="blue")
# t, = plt.plot(y_10_train[0], color="black")
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

hist = model.fit(u_10_train, y_10_train, epochs=epochs, verbose=0)

plt.figure()
loss, = plt.plot(hist.history['loss'])
logloss, = plt.plot(np.log10(hist.history['loss']))
plt.legend([loss, logloss], ["loss","log10(loss)"])

y_10_esn_train = model(u_10_train) 

print("** ploting the final results...")
plt.figure()
i, = plt.plot(u_10_train[0], color="blue")
t, = plt.plot(y_10_train[0], color="black")
o, = plt.plot(y_10_esn_train[0], color="red", linestyle='dashed')
plt.xlabel("Timesteps")
plt.legend([i, t, o], ['input', "target", "readout"])

y_10_esn_val = model(u_10_val)

plt.figure()
i, = plt.plot(u_10_val[0], color="blue")
t, = plt.plot(y_10_val[0], color="black")
o, = plt.plot(y_10_esn_val[0], color="red", linestyle='dashed')
plt.xlabel("Timesteps")
plt.legend([i, t, o], ['input', "target", "readout"])

plt.show()


