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
stime_train = 1000 # sample number for training
stime_val = 200 # sample number for validation
epochs = 200
num_units = 200 # original ESN network hidden unit number
out_plt_index = 0 # the output to be plotted
in_plt_index = 0 # the input to be plotted
sample_step = 4 # the POD sample step (in time) in MOR, smaller value means finer sampling (more samples)
order = 20 # reduced order
leaky_ratio = 1 # leaky ratio of ESN
connectivity_ratio = 0.1 # connectivity ratio of the ESN internal layer
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

# simulate the state space ESN model with training data to generate samples
y_out_train, g_sample_all, g_sample_stable_all, x_sample_all = miniesn_tools.esn_ss_sim(W, W_in, W_out, out_bias, leaky_ratio, activation_fun, u_train)

# train the original ESN
W_out = miniesn_tools.esn_train(x_sample_all[:,washout_end:], y_train[0].T[:,washout_end:])

# assign the trained W_out back to ESN
model = miniesn_tools.esn_assign(model, W_out)

# simulate the trained original ESN
y_esn_val = model(u_val)

########## construct MiniESN using the trained ESN for simulation #######################

# perform state approximation on ESN model
W_out_r, V = miniesn_gen.state_approx(W, W_in, W_out, out_bias, x_sample_all[:,washout_end:], sample_step, order)

# simulate the state approximate ESN without DEIM
y_out_sa, x_sample_sa = miniesn_tools.state_approx_sim(W, W_in, W_out_r, out_bias, V, leaky_ratio, activation_fun, u_val)

# further perform DEIM to obtain miniESN without stabilization, this model is for demonstration ONLY
W_deim, W_in_deim, E_deim, W_out_deim = miniesn_gen.miniesn_gen(W, W_in, W_out, V, g_sample_all[:,washout_end:], sample_step, order)

# simulate miniESN with DEIM using state space model, for demonstration ONLY
y_out_deim, x_sample_deim = miniesn_tools.esn_deim_sim(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, activation_fun, u_val)

# generate miniESN without stablization and assign weights, this network is for demonstration ONLY
miniesn_unstable = miniesn_tools.esn_deim_assign(E_deim, W_deim, W_in_deim, W_out_deim, out_bias, leaky_ratio, activation_fun, stime_val)

# simulate miniESN without stabilization
y_out_miniesn_unstable = miniesn_unstable(u_val)

# perform stable DEIM to get the stable miniESN
W_deim_stable, W_in_deim_stable, E_deim_stable, E_lin_stable, W_out_deim_stable = miniesn_gen.miniesn_stable(W, W_in, W_out, V, g_sample_stable_all[:,washout_end:], sample_step, order)

# simulate the stable miniESN using state space model
y_out_deim_stable, x_sample_deim_stable = miniesn_tools.esn_deim_stable_sim(E_deim_stable, E_lin_stable, W_deim_stable, W_in_deim_stable, W_out_deim_stable, out_bias, leaky_ratio, activation_fun, u_val)

########################## compute the mse errors ############################
mse_esn_org = np.mean((y_val[0, washout_end:, :] - y_esn_val[0, washout_end:, :])**2)
mse_ss_approx = np.mean((y_val[0, washout_end:, :] - y_out_sa[:,washout_end:].T)**2)
mse_miniesn_unstable = np.mean((y_val[0, washout_end:, :] - y_out_miniesn_unstable[0, washout_end:, :])**2)
mse_miniesn = np.mean((y_val[0, washout_end:, :] - y_out_deim_stable[:,washout_end:].T)**2)
print("mse_esn_org: ", mse_esn_org)
print("mse_ss_approx: ", mse_ss_approx)
print("mse_miniesn_unstable: ", mse_miniesn_unstable)
print("mse_miniesn: ", mse_miniesn)

######################### plot the accuracy comparison results ##################################

plt.figure()
t, = plt.plot(y_val[0,:,out_plt_index], color="black")
d, = plt.plot(y_out_miniesn_unstable[0,:,out_plt_index], color="magenta", linestyle='dashed')
plt.xlabel("Timesteps")
plt.legend([t, d], ["Target", "miniESN unstabilized"])

plt.figure()
t, = plt.plot(y_val[0,:,out_plt_index], color="black")
o, = plt.plot(y_esn_val[0,:,out_plt_index], color="blue", linestyle='dotted')
r, = plt.plot(y_out_sa[out_plt_index, :], color="green", linestyle='dashed')
st, = plt.plot(y_out_deim_stable[out_plt_index, :], color="red", linestyle='dashdot')
plt.xlabel("Timesteps")
plt.legend([t, o, r, st], ["Target", "ESN org", "State approx", "miniESN"])

plt.figure()
state, = plt.plot(x_sample_deim[1,:], color="black")
plt.xlabel("Timesteps")
plt.legend([state], ["state1"])

plt.show()
