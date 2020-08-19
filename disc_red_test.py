import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

a = np.array([[1, 2], [3, 4]])
b = np.array([[5], [6]])
c = np.array([[7, 8]])
d = np.array([[9]])

sys = signal.dlti(a, b, c, d)
print(sys)

w, mag, phase = sys.bode()

plt.figure()
plt.semilogx(w, mag)
plt.figure()
plt.semilogx(w, phase)
plt.show()
