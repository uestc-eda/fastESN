import numpy as np

def mor_esn(W, W_in, W_out, out_bias, sample_all, sample_step, order):
    # sample_step = x_all.shape[1]//n_sample # integer floor division "//" to compute integer sample_step
    # print("x_all=", x_all)
    print("sample_step=", sample_step)
    samples = sample_all[:, 0::sample_step]
    # print("samples=", samples)
    U, S, V = np.linalg.svd(samples, full_matrices=False)
    U = U[:,0:order]

    W_r = U.T@W@U
    W_in_r = U.T@W_in
    W_out_r = W_out@U

    return W_r, W_in_r, W_out_r, U

def miniesn_gen(W, W_in, W_out, V, sample_all, sample_step, order_deim):
    samples = sample_all[:, 0::sample_step]
    U, S, V_deim = np.linalg.svd(samples, full_matrices=False)
    U = U[:,0:order_deim]

    idx, P = deim_core(U)

    W_deim = P.T@W@V
    W_in_deim = P.T@W_in
    E_deim = V.T@U@np.linalg.inv(P.T@U)
    E_deim = E_deim.astype('float32') # convert to float to be compatible with tensorflow
    W_out_deim = W_out@V

    return W_deim, W_in_deim, E_deim, W_out_deim

def deim_core(U):
    n = U.shape[0] # original model order
    order_deim = U.shape[1] # reduced model deim order
    # U_out = np.zeros([n, order_deim])

    P = np.zeros((n, order_deim)) # initiate P matrix
    idx = np.zeros(order_deim, dtype=int) # initial index vector

    idx[0] = np.argmax(np.absolute(U[:,0]))
    # U_out[:,0] = U[:,0]
    # print("U_out=", U_out)
    P[idx[0],0] = 1 # set the first column of P

    # tmp = P[:,0:1].T@U[:,0:1]
    # print("tmp=", tmp)
    
    for i in range(1,order_deim): # loop start from the second index
        c = np.linalg.solve(P[:,0:i].T@U[:,0:i], P[:,0:i].T@U[:,i])
        # print("c=",c)
        res = U[:,i] - U[:,0:i]@c
        idx[i] = np.argmax(np.absolute(res))
        P[idx[i],i] = 1 # set the ith column of P

    # print("P=", P)
    print("idx=", idx)
    return idx, P

