'''
    Created on Oct 25th, 2020
    @authors: Alberto Chimenti, Clara Eminente and Matteo Guida.
    Purpose: (PYTHON3 IMPLEMENTATION)
        Program to create a gif of the time evolution on the Bloch sphere. 
'''

#plotting on Bloch sphere
from qutip import basis, Bloch
import matplotlib.pyplot as plt
import imageio

def qutip_qstate(coefs):
    '''
    This function creates a state to feed into qutip functions. 
    The basis is the one of one qubit.
    
    Inputs;
    #coefs: np.array(dtype=complex), array containing coeffcients of the quantum state

    Outputs:
    #QuTip state in one qubit spin basis

    '''
    up = basis(2,0)
    down = basis(2,1)
    return coefs[0]*up + coefs[1]*down


def create_gif(qstates, qstart, qtarget, name):
    '''
    Inputs:
    # qstates: list of states as np.arrays
    # qstart, qtarget: respectively the target ans start state
    # name: name of the output gif
    '''
    
    b = Bloch()
    duration=5 #framerate
    images=[]

    for (qstate,i) in zip(qstates, range(0,len(qstates))):

        b.clear()

        b.point_color = "r" # options: 'r', 'g', 'b' etc.
        b.point_marker = ['o']
        b.point_size = [40]

        b.add_states(qutip_qstate(qstart))
        b.add_states(qutip_qstate(qtarget))
        for previous in range(i):
            b.add_states(qutip_qstate(qstates[previous]),"point") #plots previous visited states as points
        b.add_states(qutip_qstate(qstate))
        filename='t.png'
        b.save(filename)
        images.append(imageio.imread(filename))

    imageio.mimsave(name, images, 'GIF', fps=duration)