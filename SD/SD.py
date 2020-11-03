from quantum_state import compute_fidelity
from Qmodel import quantum_model
import numpy as np
from random import choices
from random import uniform
from tqdm import tnrange
from copy import deepcopy


def exp_dec(iteration,percentage_flip,niterations):
    tau =  niterations/np.log(percentage_flip)
    return percentage_flip * np.exp(-iteration/tau)

def stochastic_descent(qstart, qtarget, L, T, nsteps, iterations, flips,exp_decay_flip, field_list,beta,metropolis_choice,verbose, check_norm):
    
    dt = T/nsteps
    
    #initialize model
    model=quantum_model(qstart, qtarget, dt, L, g=1, h_list=field_list, history=True)

    np.random.seed(213)

    random_protocol = np.array(choices(field_list, k=nsteps)) # Define a random protocol, sampling without replacement from a list. 
    temp_protocol=deepcopy(random_protocol)

    fidelity = model.compute_fidelity()
    fidelity_values=[fidelity]
    '''flips_iter = flips
    break_variable = 0'''

    moves = np.arange(0,nsteps,1)
    minima = False

    while not minima:

        np.random.shuffle(moves)

        for flip in moves: 

            model.reset()

            '''if exp_decay_flip is True:
                flips_iter=int(exp_dec(iteration=j,percentage_flip=flips,niterations=iterations))'''

            index_update = flip # Select an index for the update.
            temp_protocol=deepcopy(random_protocol)
            temp_protocol[index_update] = random_protocol[index_update]*(-1) # Try to update that index.
            
            evolution=model.evolve_from_protocol(temp_protocol)
            
            if check_norm and (np.abs(1 - compute_fidelity(evolution[-1],evolution[-1])) > 1e-9):
                print("Warning ---> Norm is not conserved")
                print(compute_fidelity(evolution[-1],evolution[-1]))
                break

            temp_fidelity = model.compute_fidelity() # Evaluate the fidelity
            #break_variable +=1
            if temp_fidelity>fidelity: # Update the change only if better fidelity
                random_protocol=deepcopy(temp_protocol)
                fidelity=temp_fidelity
                fidelity_values.append(fidelity)
                #break_variable = 0
                break
            #otherwise protocol is kept and we move to the next flip, unles...
            if flip==moves[-1]: #there is no more to flip
                minima=True

            '''    #best_reached_state = psi #unused but useful for debugging
            elif metropolis_choice is True:
                if np.random.uniform(0,1)<np.exp(-beta*(temp_fidelity-fidelity)):
                    random_protocol=deepcopy(temp_protocol)
                    fidelity=temp_fidelity'''

            '''fidelity_values.append(fidelity)
            if break_variable == nsteps:
                break'''
        
    
    return random_protocol, fidelity_values