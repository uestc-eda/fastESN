def mor_disc(sys, f_samples)
    n = sys.A.shape[0] # order of the system
    n_in = sys.B.shape[1] # input number of the system
    n_out = sys.C.shape[0] # output number of the system
    n_samples = f_samples.size # number of frequency sample points
    M = np.zeros(shape=(n, 2*n_samples*n_in))
    index = 0
    for f_sample in f_samples:
        m = sys.C*np.linalg.solve(exp(j*f_sample)*np.identity(n)-sys.A, sys.B) # solve system for f_sample: m = C (zI - A)^{-1} B
        M[:,2*n_in*index:2*n_in*index+n_in] = m
        M[:,2*n_in*index+n_in:2*n_in*index+2*n_in] = np.conj(m)
    U, S, V = np.linalg.svd(M, full_matrices=False)
    
