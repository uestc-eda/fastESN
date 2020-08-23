import numpy as np
import control
import matplotlib.pyplot as plt
from mor_disc import mor_disc

n_org = 10
n_in = 1
n_out = 1

omega_samples = np.array([0.1, 1.4, 2.4])

a = np.random.rand(n_org, n_org)
b = np.random.rand(n_org, n_in)
c = np.random.rand(n_out, n_org)
d = np.random.rand(n_out, n_in)

sys = control.ss(a, b, c, d, 1)
print(sys)

sys_r = mor_disc(sys, omega_samples)
print(sys_r)


in_plot = 0 # the input index (starting from 0) of the frequence response to be ploted
out_plot = 0 # the output index (starting from 0) of the frequence response to be ploted
sys_plot = control.ss(sys.A, sys.B[:, [in_plot]], sys.C[[out_plot], :], sys.D[[out_plot], [in_plot]], 1) # the SISO system (extracted from the MIMO system) to be ploted
print(sys_plot)

sys_r_plot = control.ss(sys_r.A, sys_r.B[:, [in_plot]], sys_r.C[[out_plot], :], sys_r.D[[out_plot], [in_plot]], 1)

mag, phase, w = control.bode(sys_plot)

mag_r, phase_r, w_r = control.bode(sys_r_plot)
# (mag, phase, freq) = sys.freqresp(freq)
# plt.figure()
# plt.semilogx(w, mag)
# plt.figure()
# plt.semilogx(w, phase)
plt.show()
