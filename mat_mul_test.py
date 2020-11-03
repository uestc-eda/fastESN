# import os
# os.environ.update(
#     OMP_NUM_THREADS = '1',
#     OPENBLAS_NUM_THREADS = '1'
#     # NUMEXPR_NUM_THREADS = '1'
#     # MKL_NUM_THREADS = '1'
# )

import numpy as np
import time

# mat_size_set = list(range(10, 100, 10))
mat_size_set = [2, 10, 50, 90, 100, 1000]

for i in range(0, len(mat_size_set)):
    mat_size = mat_size_set[i]
    A = np.random.rand(mat_size, mat_size)
    B = np.random.rand(mat_size, 1)
    C = np.zeros((mat_size, 1))
    # t = time.process_time()
    t = time.perf_counter()
    for j in range(0, 10000):
        C = A@B
    # C = A@B
    # mul_time = time.process_time() - t
    mul_time = time.perf_counter() - t
    print("mat_size: ", mat_size)
    print("mul_time: ", mul_time)
    
