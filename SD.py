'''
    Created on Oct 25th, 2020
    @authors: Alberto Chimenti, Clara Eminente and Matteo Guida.
    Purpose: (PYTHON3 IMPLEMENTATION)
        Methods and class to instantiate and manipulate both single qubits and many-quntum body pure and separable systems.
'''

from Qmodel import quantum_model
import numpy as np
from random import choices
from random import uniform
from tqdm import tnrange
from itertools import combinations
from copy import deepcopy

def compute_fidelity(target, psi):
    '''
    Given two pure quantum states, target and psi, the fidelity is return, i.e. their closeness in Hilbert space.
    '''
    F = np.abs(np.vdot(target, psi))**2
    return F

def correlation(matrix,h):
    '''
    Given a matrix this function computes the correlation quantity q(T) as in the paper PhysRevX.8.031086 pag. 5.
    Matrix must have dimension (n_protocols,lenght_of_protocol).    
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

def stochastic_descent(qstart, qtarget, L, T, nsteps,nflip,field_list, verbose, check_norm):
    ''' 
    The function perform stochastic descent algorithm for a system of dimension L from an initial state qstart to reach the final state qtarget
    starting from a random protocol with total duration T and sequence of lenght nstep of values of the control field among the possible values given by
    the field list. The algorithm stops when an entire flip_list is covered without any increase of the fidelity and the obtained fidelity is larger than 
    the starting one between qstart and qtarget. 
    It returns the best protocol found and a list with each increase of the fidelity. 
    '''
    # dt of the evolution for each element value of the protocol.  
    dt = T/nsteps
    
    # Initialize model.
    model=quantum_model(qstart, qtarget, dt, L, g=1, h_list=field_list, history=True)

    np.random.seed(213)
    # Define a random protocol, sampling without replacement from a list. 
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

        deb_index = 0
        for flip in moves: 
            deb_index += 1
            model.reset()

            # Select an index for the update.
            index_update = flip_list[flip] 
            temp_protocol=deepcopy(random_protocol)
            # Try to update that/those index/indices in the protocol.
            temp_protocol[index_update] = random_protocol[index_update]*(-1) 
            # Compute time evolution according with the previous updated protocol.
            evolution=model.evolve_from_protocol(temp_protocol)
            
            # Norm check after the evolution for the sake of debugging.
            if check_norm and (np.abs(1 - compute_fidelity(evolution[-1],evolution[-1])) > 1e-9):
                print("Warning ---> Norm is not conserved")
                print(compute_fidelity(evolution[-1],evolution[-1]))
                break
            
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