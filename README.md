# Model-Based Meta-Policy Optimization (MB-MPO)

This repository contains code corresponding to the paper ["Model-Based Meta-Policy Optimization"](https://arxiv.org/abs/1809.05214). 

### Dependencies
This code is based off of the [rllab](https://github.com/rll/rllab) code repository as well as the [maml_rl](https://github.com/cbfinn/maml_rl) repository and can be installed in the same way (see below).
This codebase is not necessarily backwards compatible with rllab. The code uses the TensorFlow rllab version which can be find in the folder sandbox, so be sure to install TensorFlow v1.0+.
Furthermore baseline inplementations of PPO, ACKTR and DDPG from [open-ai baselines](https://github.com/openai/baselines) are also included in the paper


### Installation

To install all neccessary packages and dependencies, please follow th instructions on the [rllab documentation](https://rllab.readthedocs.io/en/latest/user/installation.html#express-install).
Also be aware that for running the experiments, the [Mujoco physics simulator](http://www.mujoco.org/) 1.3 is required, which requires a licence.


### Usage

The core components of our code such as the algorithm can be found in the directory `sandbox/ours/`.

Scripts for running the experiments found in the paper are located in `experiments/run_scripts`.
For each experiment in the paper a corresponding folder in `experiments/run_scripts` contains the runscripts.

For instance, in order to run MB-MPO on your local machine execute the folloowing command from the root of this repository:

`python experiments/run_scripts/mb_mpo_train.py --mode local`

The hyperparameters and the environment(s) on which to run the experiments can be specified in the same file.

The results and logs of the experiment run are saved into the folder `data/local/`.

## rllab

rllab is a framework for developing and evaluating reinforcement learning algorithms. It includes a wide range of continuous control tasks plus implementations of the following algorithms:


- [REINFORCE](https://github.com/rllab/rllab/blob/master/rllab/algos/vpg.py)
- [Truncated Natural Policy Gradient](https://github.com/rllab/rllab/blob/master/rllab/algos/tnpg.py)
- [Reward-Weighted Regression](https://github.com/rllab/rllab/blob/master/rllab/algos/erwr.py)
- [Relative Entropy Policy Search](https://github.com/rllab/rllab/blob/master/rllab/algos/reps.py)
- [Trust Region Policy Optimization](https://github.com/rllab/rllab/blob/master/rllab/algos/trpo.py)
- [Cross Entropy Method](https://github.com/rllab/rllab/blob/master/rllab/algos/cem.py)
- [Covariance Matrix Adaption Evolution Strategy](https://github.com/rllab/rllab/blob/master/rllab/algos/cma_es.py)
- [Deep Deterministic Policy Gradient](https://github.com/rllab/rllab/blob/master/rllab/algos/ddpg.py)

rllab is fully compatible with [OpenAI Gym](https://gym.openai.com/). See [here](http://rllab.readthedocs.io/en/latest/user/gym_integration.html) for instructions and examples.

rllab only officially supports Python 3.5+. For an older snapshot of rllab sitting on Python 2, please use the [py2 branch](https://github.com/rllab/rllab/tree/py2).

rllab comes with support for running reinforcement learning experiments on an EC2 cluster, and tools for visualizing the results. See the [documentation](https://rllab.readthedocs.io/en/latest/user/cluster.html) for details.

The main modules use [Theano](http://deeplearning.net/software/theano/) as the underlying framework, and we have support for TensorFlow under [sandbox/rocky/tf](https://github.com/openai/rllab/tree/master/sandbox/rocky/tf).

### Documentation

Documentation is available online: [https://rllab.readthedocs.org/en/latest/](https://rllab.readthedocs.org/en/latest/).

#$ Citing rllab

If you use rllab for academic research, you are highly encouraged to cite the following paper:

- Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel. "[Benchmarking Deep Reinforcement Learning for Continuous Control](http://arxiv.org/abs/1604.06778)". _Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016._

### Credits

rllab was originally developed by Rocky Duan (UC Berkeley / OpenAI), Peter Chen (UC Berkeley), Rein Houthooft (UC Berkeley / OpenAI), John Schulman (UC Berkeley / OpenAI), and Pieter Abbeel (UC Berkeley / OpenAI). The library is continued to be jointly developed by people at OpenAI and UC Berkeley.

### Slides

Slides presented at ICML 2016: https://www.dropbox.com/s/rqtpp1jv2jtzxeg/ICML2016_benchmarking_slides.pdf?dl=0

#
