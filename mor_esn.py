import numpy as np

def mor_esn(W, W_in, W_out, out_bias, x_all, n_sample, order):
    sample_step = x_all.shape[1]//n_sample # integer floor division "//" to compute integer sample_step
    print(sample_step)
    samples = x_all[:, 0::sample_step]
    print(samples.shape)
    U, S, V = np.linalg.svd(samples, full_matrices=False)
    U = U[:,0:order]

    W_r = U.T@W@U
    W_in_r = U.T@W_in
    W_out_r = W_out@U

    return W_r, W_in_r, W_out_r, U
