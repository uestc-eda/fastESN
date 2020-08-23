import numpy as np
import control

def mor_disc(sys, omega_samples):
    n = sys.A.shape[0] # order of the system
    n_in = sys.B.shape[1] # input number of the system
    n_out = sys.C.shape[0] # output number of the system
    n_samples = omega_samples.size # number of frequency sample points
    
    M = np.zeros(shape=(n, 2*n_samples*n_in), dtype=complex)
    index = 0
    for omega_sample in omega_samples:
        m = np.linalg.solve(np.exp(1j*omega_sample)*np.identity(n)-sys.A, sys.B) # solve system for omega_sample: m = (zI - A)^{-1} B
        M[:,2*n_in*index:2*n_in*index+n_in] = m
        M[:,2*n_in*index+n_in:2*n_in*index+2*n_in] = np.conj(m)
        index = index + 1

    print(M)
    
    U, S, V = np.linalg.svd(M, full_matrices=False)

    # convert U to real matrix by angle rotation
    u_angle = np.angle(U[0,:])
    for i in range(U.shape[1]):
        U[:,i] = U[:,i]*np.exp(-1j*u_angle[i])
    U = U.real

    Ar = U.T@sys.A@U
    Br = U.T@sys.B
    Cr = sys.C@U
    Dr = sys.D

    sys_r = control.ss(Ar, Br, Cr, Dr, 1)

    return sys_r
