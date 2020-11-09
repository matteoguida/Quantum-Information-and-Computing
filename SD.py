'''
    Created on Oct 25th, 2020
    @authors: Matteo Guida, Clara Eminente, Alberto Chimenti.
    Purpose: (PYTHON3 IMPLEMENTATION)
        Stochastic descent method with tunable number of flip to aim to find an optimal protocol starting from a random one.
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
    Given a matrix this function computes the correlation quantity Q(T) as in the paper.
    Matrix must have dimension (n_protocols,lenght_of_protocol)    
    '''
    n_row = matrix.shape[0]
    n_col = matrix.shape[1] 
    
    mean_hx = np.array([matrix[:,i].mean() for i in range(n_col)]) #mean over all protocols at fixed time
    avg_over_h = np.array([np.array([ (matrix[i,j]-mean_hx[j])**2 for i in range(n_row)]).sum()/n_row for j in range(n_col)])
    avg_over_Nt = (1/((h*h)*n_col))*(avg_over_h.sum())
    return  avg_over_Nt

def stochastic_descent(qstart, qtarget, L, T, nsteps,nflip,field_list, verbose, check_norm):
    
    dt = T/nsteps
    
    #initialize model
    model=quantum_model(qstart, qtarget, dt, L, g=1, h_list=field_list, history=True)

    np.random.seed(213)

    random_protocol = np.array(choices(field_list, k=nsteps)) # Define a random protocol, sampling without replacement from a list. 
    temp_protocol=deepcopy(random_protocol)

    start_fidelity = model.compute_fidelity()
    fidelity = deepcopy(start_fidelity)
    fidelity_values=[start_fidelity]

    flip_list = [i for i in range(nsteps)]
    index = np.arange(0,nsteps,1)
    for s in range(1, nflip):
        for e in combinations(index,s+1):
            flip_list.append(list(e))
    move = len(flip_list)
    moves = np.arange(0, move, dtype=int)
    

    minima = False

    while not minima:

        np.random.shuffle(moves)

        deb_index = 0
        for flip in moves: 
            deb_index += 1
            model.reset()

            index_update = flip_list[flip] # Select an index for the update.
            temp_protocol=deepcopy(random_protocol)
            temp_protocol[index_update] = random_protocol[index_update]*(-1) # Try to update that index.
            
            evolution=model.evolve_from_protocol(temp_protocol)
            
            # Norm check after the evolution for the sake of debugging.
            if check_norm and (np.abs(1 - compute_fidelity(evolution[-1],evolution[-1])) > 1e-9):
                print("Warning ---> Norm is not conserved")
                print(compute_fidelity(evolution[-1],evolution[-1]))
                break
            
            temp_fidelity = model.compute_fidelity() 
        
            if temp_fidelity > fidelity: # Update the change only if better fidelity
                random_protocol=deepcopy(temp_protocol)
                fidelity=temp_fidelity
                fidelity_values.append(fidelity)
                break
            
            
            #otherwise protocol is kept and we move to the next flip, unles...
            if flip==moves[-1] and temp_fidelity>start_fidelity: #there is no more to flip
                minima=True
            elif  flip==moves[-1] and temp_fidelity<start_fidelity:
                random_protocol = np.array(choices(field_list, k=nsteps))    
    return random_protocol, fidelity_values