import numpy as np

def state_approx(W, W_in, W_out, out_bias, sample_all, sample_step, order):
    # sample_step = x_all.shape[1]//n_sample # integer floor division "//" to compute integer sample_step
    # print("x_all=", x_all)
    print("sample_step=", sample_step)
    samples = sample_all[:, 1::sample_step]

    U_right, S_right, V_right = np.linalg.svd(samples, full_matrices=False)
    U_right = U_right[:,0:order]

    U_left = U_right # we use U_left as U_right, they can be different but more problematic

    W_out_r = W_out@U_right

    return W_out_r, U_left, U_right

def miniesn_gen(W, W_in, W_out, V_left, V_right, sample_all, sample_step, order_deim):
    samples = sample_all[:, 1::sample_step]
    U, S, V_deim = np.linalg.svd(samples, full_matrices=False)
    U = U[:,0:order_deim]

    idx, P = deim_core(U)
    # idx, P = greedy_core(U)

    W_deim = P.T@W@V_right
    W_in_deim = P.T@W_in
    E_deim = np.linalg.solve(U.T@P, U.T@V_left).T # in original math: E_deim = V_left.T@U@np.linalg.inv(P.T@U)
    
    # E_deim_norm = np.linalg.norm(E_deim, ord=2)
    # print("E_deim_norm: ", E_deim_norm)
    # PTU_norm = np.linalg.norm(P.T@U, ord=2)
    # print("PTU_norm: ", PTU_norm)
    # PTU_inv_norm = np.linalg.norm(np.linalg.inv(P.T@U), ord=2)
    # print("PTU_inv_norm: ", PTU_inv_norm)
    # VTU_norm = np.linalg.norm(V.T@U, ord=2)
    # print("VTU_norm: ", VTU_norm)
    # W_norm = np.linalg.norm(W, ord=2)
    # print("W_norm: ", W_norm)
    # W_deim_norm = np.linalg.norm(W_deim, ord=2)
    # print("W_deim_norm: ", W_deim_norm)
    # WV_norm = np.linalg.norm(W@V, ord=2)
    # print("WV_norm: ", WV_norm)
    
    # E_deim = E_deim/E_deim_norm # normalize E_deim, to keep the echo property
    E_deim = E_deim.astype('float32') # convert to float to be compatible with tensorflow
    W_out_deim = W_out@V_right
    # W_out_deim = W_out_deim.astype('float32') # convert to float to be compatible with tensorflow
    # W_out_deim = E_deim_norm*W_out@V # E_deim_norm is multiplied here because E_deim is normalized

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

    # print("U=", U)
    
    for i in range(1,order_deim): # loop start from the second index
        c = np.linalg.solve(P[:,0:i].T@U[:,0:i], P[:,0:i].T@U[:,i])
        # print("c=",c)
        res = U[:,i] - U[:,0:i]@c
        # print("res=",res)
        idx[i] = np.argmax(np.absolute(res))
        P[idx[i],i] = 1 # set the ith column of P
        # PTU_inv = np.linalg.inv(P[:,0:i].T@U[:,0:i])
        # print("PTU_inv=", PTU_inv)
        # PTU_inv_norm = np.linalg.norm(PTU_inv, ord=2)
        # print("PTU_inv_norm: ", PTU_inv_norm)
        
        # c_after = np.linalg.solve(P[:,0:i+1].T@U[:,0:i+1], P[:,0:i+1].T@U[:,i])
        # res_after = U[:,i] - U[:,0:i+1]@c_after
        # print("res_after=",res_after)
        
    # c = np.linalg.solve(P[:,0:order_deim].T@U[:,0:order_deim], P[:,0:order_deim].T@U[:,order_deim-1])
    # res = U[:,order_deim-1] - U[:,0:order_deim]@c
    # print("res=",res)
    # print("U=", U)
    # print("P=", P)
    # print("idx=", idx)
    return idx, P

def miniesn_stable(W, W_in, W_out, V_left, V_right, sample_all, sample_step, order_deim):
    samples = sample_all[:, 1::sample_step]
    U, S, V_deim = np.linalg.svd(samples, full_matrices=False)
    U = U[:,0:order_deim]

    # # use qr to orthogonolize U and V against each other
    # UV = np.concatenate((U, V), axis=1)
    # print("UV_shape: ", UV.shape)
    # UV, R = np.linalg.qr(UV, mode='reduced') # R will not be used
    # print("UV_orth_shape: ", UV.shape)
    # U = UV[:,:order_deim]
    # V = UV[:,order_deim:]
    # print("U_shape: ", U.shape)
    # print("V_shape: ", V.shape)

    idx, P = deim_core(U)
    # idx, P = greedy_core(U)

    W_deim = P.T@W@V_right
    W_in_deim = P.T@W_in
    E_deim = np.linalg.solve(U.T@P, U.T@V_left).T # in original math: E_deim = V_left.T@U@np.linalg.inv(P.T@U)
    E_lin = V_left.T@W@V_right-E_deim@P.T@W@V_right
    
    # E_deim = E_deim/E_deim_norm # normalize E_deim, to keep the echo property
    E_deim = E_deim.astype('float32') # convert to float to be compatible with tensorflow
    # E_lin = E_lin.astype('float32')
    W_out_deim = W_out@V_right
    # W_out_deim = E_deim_norm*W_out@V # E_deim_norm is multiplied here because E_deim is normalized

    return W_deim, W_in_deim, E_deim, E_lin, W_out_deim

