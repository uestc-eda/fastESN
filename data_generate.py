import numpy as np

def narma_30_gen(n_t):
    y = np.zeros(n_t+31) # initiate the output data
    u = 0.5*np.random.random(n_t+30) # input data, randomly sampled in [0, 0.5)
    # print("u = ", u)
    for t in range(30,n_t+30):
        y[t+1] = 0.2*y[t] + 0.04*y[t] * np.sum(y[t-29:t]) + 1.5*u[t-29]*u[t] + 0.001
    y = y[30:-1] # truncate the first 30 data because they are zeros to initiate the time series, also ignore the last element to agree with the inputs
    u = u[30:] # truncate the first 30 data
    # print("y = ", y)
    y = y.astype('float32') # convert to float32 to be compatible with keras
    u = u.astype('float32')
    return y, u
    
def narma_10_gen(n_t):
    y = np.zeros(n_t+11) # initiate the output data
    u = 0.5*np.random.random(n_t+10) # input data, randomly sampled in [0, 0.5)
    # print("u = ", u)
    for t in range(10,n_t+10):
        y[t+1] = 0.3*y[t] + 0.05*y[t] * np.sum(y[t-9:t]) + 1.5*u[t-9]*u[t] + 0.1
    y = y[10:-1] # truncate the first 10 data because they are zeros to initiate the time series, also ignore the last element to agree with the inputs
    u = u[10:] # truncate the first 10 data
    # print("y = ", y)
    y = y.astype('float32') # convert to float32 to be compatible with keras
    u = u.astype('float32')
    return y, u
    
