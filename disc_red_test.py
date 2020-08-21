import numpy as np
import control
import matplotlib.pyplot as plt

n_org = 10
n_in = 2
n_out = 2

a = np.random.rand(n_org, n_org)
b = np.random.rand(n_org, n_in)
c = np.random.rand(n_out, n_org)
d = np.random.rand(n_out, n_in)

sys = control.ss(a, b, c, d, 1)
print(sys)



in_plot = 0 # the input index (starting from 0) of the frequence response to be ploted
out_plot = 0 # the output index (starting from 0) of the frequence response to be ploted
sys_plot = control.ss(a, b[:, [in_plot]], c[[out_plot], :], d[[out_plot], [in_plot]], 1) # the SISO system (extracted from the MIMO system) to be ploted
print(sys_plot)

mag, phase, w = control.bode(sys_plot)
# (mag, phase, freq) = sys.freqresp(freq)
# plt.figure()
# plt.semilogx(w, mag)
# plt.figure()
# plt.semilogx(w, phase)
plt.show()
