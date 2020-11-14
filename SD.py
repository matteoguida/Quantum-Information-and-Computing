'''
    Created on Oct 25th, 2020
    @authors: Alberto Chimenti, Clara Eminente and Matteo Guida.
    Purpose: (PYTHON3 IMPLEMENTATION)
        Methods and class to instantiate and manipulate both single qubits and many-quntum body pure and separable systems.
'''

from Qmodel import quantum_model, compute_fidelity_ext
import numpy as np
from random import choices
from random import uniform
from tqdm import tnrange
from itertools import combinations
from copy import deepcopy

def correlation(matrix,h):
    '''
    Given a matrix this function computes the correlation quantity q(T) as in the paper PhysRevX.8.031086 pag. 5.
    Matrix must have dimension (n_protocols,lenght_of_protocol). Intries in the matrix have values = {-h,h}    

    INPUTS:
    matrix: np.arra() of size (n_protocols,lenght_of_protocol)
    h: float, absolute value of the field (for normalization purpose)

    OUTPUTS:
    avg_over_Nt( is q(T) ): float, between 0 and 1

    '''
    n_row = matrix.shape[0]
    n_col = matrix.shape[1] 

    # Mean over all protocols at fixed time.
    mean_hx = np.array([matrix[:,i].mean() for i in range(n_col)]) 
    # Array with entries the different addends of the sum in the q(T) formula. 
    avg_over_h = np.array([np.array([ (matrix[i,j]-mean_hx[j])**2 for i in range(n_row)]).sum()/n_row for j in range(n_col)])
    # Performing the sum. 
    avg_over_Nt = (1/((h*h)*n_col))*(avg_over_h.sum())
    return  avg_over_Nt




def stochastic_descent(qstart, qtarget, L, T, nsteps, nflip, field_list):
    
    ''' 
    The function performs stochastic descent for a system of dimension L from an initial state qstart to reach the final state qtarget
    starting from a random protocol with total duration T and nsteps steps, The values of the protocol al the subsequent values of the control field 
    and take values in field_list. The algorithm stops when an entire flip_list is covered without any increase of the fidelity and the obtained 
    fidelity is larger than the starting one between qstart and qtarget. 
    It returns the best protocol found and a list with each increase of the fidelity. 

    INPUTS:
    qstart, qtarget: np.array(dtype=complex) of size 2^L, respectively the initial, target quantum states
    L: integer >0, number of Qubits
    T: float, duration of the protocol
    nsteps: integer, steps in the protocol
    nflip: integer, maximum number of flips at a time
    field_list: list of float, list of all possible field values to precompute and store eigenvalues and eignvectors of the corresponding hamiltonians


    OUTPUTS:
    random_protocol: np.array() of size nsteps, protocol corresponding to the best achieved fidelity
    fidelity_values: list, log of updates in fidelity during the descent

    '''

    # dt of the evolution for each element value of the protocol.  
    dt = T/nsteps
    
    # Initialize model.
    model=quantum_model(qstart, qtarget, dt, L, g=1, h_list=field_list, history=True)

    np.random.seed(213)
    # Define a random protocol, sampling from a list. 
    random_protocol = np.array(choices(field_list, k=nsteps)) 
    # Make a copy with the appropriate function s.t. a future change in temp_protocol does not effect random_protocol.
    temp_protocol=deepcopy(random_protocol)


    start_fidelity = model.compute_fidelity()
    fidelity = deepcopy(start_fidelity)
    fidelity_values=[start_fidelity]


    flip_list = [i for i in range(nsteps)]
    # List with index of protocol array. 
    index = np.arange(0,nsteps,1)
    # If we perform more than a flip at each iteration, in the flip_list beyond the element corresponding to the single index of the protocol array
    # all the combinations possible for the given number nflip of the indices and for lower combinations up to one are created. 
    for s in range(1, nflip):
        for e in combinations(index,s+1):
            flip_list.append(list(e))
    move = len(flip_list)
    # Array with the indices of flip_list.
    moves = np.arange(0, move, dtype=int)
    
    # Boolean variable to stop the while.
    minima = False

    while not minima:

        # Randomly shuffle the array with the indices of flip_list.
        np.random.shuffle(moves)

        for flip in moves: 
            # Everytime we try a flip the model has to reinitialized in order to start evolution from qstart
            model.reset()
            # Select an index for the update.
            index_update = flip_list[flip] 
            temp_protocol=deepcopy(random_protocol)
            # Try to update that/those index/indices in the protocol.
            temp_protocol[index_update] = random_protocol[index_update]*(-1) 
            # Compute time evolution according with the previous updated protocol.
            evolution=model.evolve_from_protocol(temp_protocol)
            
            temp_fidelity = model.compute_fidelity() 

            # Keep the change in the protocol only if it determines better fidelity, in this case the fidelity is stored in fidelity_values. 
            if temp_fidelity > fidelity: 
                random_protocol=deepcopy(temp_protocol)
                fidelity=temp_fidelity
                fidelity_values.append(fidelity)
                break
            
            
            # Otherwise the "old" protocol is kept and we move to the next flip/s, unless two conditions are met:
            # 1) the entire list of "flip indices" is covered and so we are in a minimum 2) condition 1) is satisfied but the obtained fidelity 
            # is lower than the starting one and awful random protocol was extracted at the beginning and for this reason it is extracted again. 
            if flip==moves[-1] and temp_fidelity>start_fidelity:
                minima=True
            elif  flip==moves[-1] and temp_fidelity<start_fidelity:
                random_protocol = np.array(choices(field_list, k=nsteps))    
    return random_protocol, fidelity_values