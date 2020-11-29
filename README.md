# QUANTUM INFORMATION AND COMPUTING
![](https://media.giphy.com/media/Hs7QOaQHyo9qhwmv4h/giphy.gif)


### Prerequisites
Python versions supported:

[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://badge.fury.io/py/root_pandas)

### Authors:

- [Alberto Chimenti](https://github.com/albchim) (University of Padova)
- [Clara Eminente](https://github.com/ceminente) (University of Padova)
- [Matteo Guida](https://github.com/matteoguida) (University of Padova)

# Goal and Results 
Controlling non-integrable many-body quantum systems of interacting qubits is crucial in many areas of physics and in particular in quantum information science. In the following work a **Reinforcement Learning** (RL) algorithm is implemented in order to find an **optimal protocol** that drives a quantum system from an initial to a target state in two study cases: a single isolated qubit and a closed chain of **L coupled qubits**. For both cases the obtained results are compared with the ones achieved through **Stochastic Descent** (SD). 
What has been found is that, for a single qubit, both methods find optimal protocols whenever the total protocol duration T allows it. When the number of qubits increases RL turns out to be more flexible and to require less tuning in order to find better solutions. We also find that both algorithms capture the role of T. 

A complete explanation of the results and the code development can be found in Report.pdf.

The work is based on some of the results obtained in [1]

#### RL and SD Usage:
In order to run the training just run:

  >python RL_training.py

To run SD:

  >python script_SD.py
  
In both the program use flag -h or --help to print a brief description of the script and useful informations about init parameters.

### Useful External Links:
[1] Bukov,   A.  G.  R.  Day,   D.  Sels,   P.  Weinberg,  A.  Polkovnikov,  and  P.  Mehta,  [Reinforcement learning in different phases of quantum control](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031086), Phys. Rev. X8, 031086 (2018). 

[2] S. Montangero, [Introduction to Tensor Network Methods: Numerical simulations of low-dimensional many body quantum systems](https://www.springer.com/gp/book/9783030014087) (Springer Nature Switzerland AG, 2018).
