# Quantum-Computing

<img src=https://www.researchgate.net/publication/335028508/figure/fig1/AS:789466423762944@1565234871365/The-Bloch-sphere-provides-a-useful-means-of-visualizing-the-state-of-a-single-qubit-and.ppm width="500" height="450" border="0"/> 

### Prerequisites
Python versions supported:

[![](https://img.shields.io/badge/python-3.7.9-blue.svg)](https://badge.fury.io/py/root_pandas)

### Installing

You can recreate the conda environment used for this work with:

```
conda env create -f environment.yml
```

###  Description and Results
Controlling non-integrable many-body quantum systems of interacting qubits is crucial in many areas of physicsand  in  particular  in  quantum  information  science.   In  this  work  a  Reinforcement  Learning  (RL) algorithm is implemented in order to find an optimal protocol that drives a quantum system from an initialto a target state in two study cases:  a single isolated qubit and a closed chain of L coupled qubits.  For bothcases the obtained results are compared with the ones achieved through Stochastic Descent (SD). What is found is that, for a single qubit, both methods find optimal protocols whenever the total protocol duration T allows it.  When the number of qubits increases RL turns out to bemore flexible and to require less work and tuning in order to find the best achievable solutions.  We also findthat both algorithms capture the role ofT.  The work is based on some of the results obtained in [1] An exhaustive explanation of the used RL algorithm and other details related to code development and discussion of the obtained results can be found in REPORT.pdf.
### Authors:

- [Alberto Chimenti](https://github.com/albchim) (University of Padova)
- [Clara Eminente](https://github.com/ceminente) (University of Padova)
- [Matteo Guida](https://github.com/matteoguida) (University of Padova)

### Useful External Links:
[1] [M. Bukov, A. G. R. Day, D. Sels, P. Weinberg, A. Polkovnikov, and P. Mehta, Reinforcement learning indifferent phases of quantum control,Phys. Rev. X8, 031086 (2018).](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031086)
