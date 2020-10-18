'''
    Created on Oct 18, 2020

    @author: Alberto Chimenti

    Purpose: (PYTHON3 IMPLEMENTATION)
        Implementation of some qbit state manipulation functions
'''

### To read for further implementation:
# https://www.researchgate.net/publication/47744360_Spin_operator_matrix_elements_in_the_quantum_Ising_chain_Fermion_approach

#%%
import numpy as np
'''
    Global variable for easier use (BEWARE OF LOOP INDICES CHOICE)
'''
i = 0. + 1.j

def hamiltonian(field, single=True):
    '''
        Flexible implementation of naive ising hamiltonian
    '''
    if single:
        H=np.array([[field, 1],[1, -field]], dtype=complex)
        return(H)
    else:
        pass


def time_evolution(psi, dt, field, trotter=True):
    '''
        Flexible implementation of time evolution of pure quantum states
        for a variable number of qbits
        --> To implement other options than Trotter-Suzuki decomposition for time
            evolution unitary operator
    '''
    if trotter:
        identity = np.diag(np.ones(len(psi)))
        psi = np.dot((identity -i*hamiltonian(field)*dt), psi)
        return(psi)
    else:
        pass


if __name__ == "__main__":

    H = hamiltonian(-2)
    print(H)

    time_evolution([1,1], 0.05, -2)