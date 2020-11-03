from Qmodel import quantum_model
import numpy as np
from random import choices
from random import uniform
from tqdm import tnrange
from copy import deepcopy

def compute_fidelity(target, psi):
    F = np.abs(np.vdot(target, psi))**2
    return F

def exp_dec(iteration,percentage_flip,niterations):
    tau =  niterations/np.log(percentage_flip)
    return percentage_flip * np.exp(-iteration/tau)

def stochastic_descent(qstart, qtarget, L, T, dt, flips , exp_decay_flip, field_list, beta, metropolis_choice,verbose, check_norm):
    
    nsteps = int(T/dt)
    
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
    '''from itertools import combinations
    idx = np.arange(0,nsteps,dtype=int)
    flip_list = [i for i in range(nsteps)] # list of idx
    for s in range(1, 2):
        for e in combinations(idx,s+1):
            flip_list.append(list(e))
    moves=flip_list'''


    minima = False


    while not minima:

        np.random.shuffle(moves)


        for flip in moves: 

            #break_var+=1

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

            temp_fidelity = model.compute_fidelity() 
        
            if temp_fidelity > fidelity: # Update the change only if better fidelity
                random_protocol=deepcopy(temp_protocol)
                fidelity=temp_fidelity
                fidelity_values.append(fidelity)
                break
            '''elif metropolis_choice is True:
                rand = np.random.uniform(0,1)
                met_weight = np.exp(beta*(temp_fidelity-fidelity))
                if rand>met_weight:
                    #print(break_var)
                    #print("Metropolis accepted:", temp_fidelity, fidelity, rand, met_weight)
                    random_protocol=deepcopy(temp_protocol)
                    fidelity=temp_fidelity
                    break'''

            #otherwise protocol is kept and we move to the next flip, unles...
            if flip==moves[-1] : #there is no more to flip
                #print("DEBUGGING")
                minima=True


            '''    #best_reached_state = psi #unused but useful for debugging
            '''

            '''fidelity_values.append(fidelity)
            if break_variable == nsteps:
                break'''
        
    
    return random_protocol, fidelity_values