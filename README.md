# fastESN: fast echo state network (ESN)

This repository contains code of fastESN, published in

> H. Wang, X. Long, X.-X. Liu, **"[fastESN: Fast Echo State Network](https://wanghaiuestc.github.io/papers/tnnls_fastesn_author_copy.pdf)."** *IEEE Transactions on Neural Networks and Learning Systems (TNNLS)*, vol. 34, no. 12, December 2023, pp. 10487-10501.

From a trained ESN model, fastESN creates a smaller echo state network (ESN) model of user defined size, at runtime speed (only in mini seconds). As a result, simulation using fastESN can be much faster than the original ESN.

See [here](https://wanghaiuestc.github.io) for more opensource softwares from my group. 

## Installation

1. Create and activate a virtual enviroment, using anaconda, venv, ect.

2. Install tensorflow, scipy, matplotlib

## Run

Run ```simulation_mode_demo.py``` to view the results. Change settings at the beginning of this script, if you want to see more than the default.

```sh
python3 ./simulation_mode_demo.py
```


The statistical results of our paper are reported by running the script ```data_collect_simulation.py```, for different model sizes and sample numbers.

```sh
python3 ./data_collect_simulation.py
```