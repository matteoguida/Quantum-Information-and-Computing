from Qmodel import quantum_model, compute_fidelity_ext
import numpy as np
from random import choices
from random import uniform
from tqdm import tnrange
from copy import deepcopy


def stochastic_descent(qstart, qtarget, L, T, nsteps, field_list, verbose, check_norm):
    
    dt = T/nsteps
    
    #initialize model
    model=quantum_model(qstart, qtarget, dt, L, g=1, h_list=field_list, history=True)

    np.random.seed(213)

    random_protocol = np.array(choices(field_list, k=nsteps)) # Define a random protocol, sampling without replacement from a list. 
    temp_protocol=deepcopy(random_protocol)

    fidelity = model.compute_fidelity()
    fidelity_values=[fidelity]
    start_fidelity=deepcopy(fidelity)

    moves = np.arange(0,nsteps,1)

    minima = False
    
    while not minima:

        np.random.shuffle(moves)

        for flip in moves: 

            model.reset()

            index_update = flip # Select an index for the update.
            temp_protocol=deepcopy(random_protocol)
            temp_protocol[index_update] = random_protocol[index_update]*(-1) # Try to update that index.
            
            evolution=model.evolve_from_protocol(temp_protocol)
            
            temp_fidelity = model.compute_fidelity() 
            
            if temp_fidelity > fidelity: # Update the change only if better fidelity
                random_protocol=deepcopy(temp_protocol)
                fidelity=temp_fidelity
                fidelity_values.append(fidelity)
                break
           
            #otherwise protocol is kept and we move to the next flip, unles...
            if flip==moves[-1] and temp_fidelity>start_fidelity: #there is no more to flip
                minima=True
            elif  flip==moves[-1] and temp_fidelity<start_fidelity: #if he found a very bad protocol lets start again and look for an even slightly better one
                random_protocol = np.array(choices(field_list, k=nsteps))
      
    
    return random_protocol, fidelity_values