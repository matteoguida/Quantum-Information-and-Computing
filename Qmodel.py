'''
    Created on Oct 25th, 2020
    @authors: Alberto Chimenti, Clara Eminente and Matteo Guida.
    Purpose: (PYTHON3 IMPLEMENTATION)
        Methods and class to instantiate and manipulate both single qubits and many-quntum body pure and separable systems.
'''

#%%
from profiler_decorator import profile
import numpy as np
import copy


def compute_H_and_LA(L, g, field):
    from numpy import linalg as LA
    #1/2 spin operators.
    sigma_x=1/2*np.array([[0,1],[1,0]])
    sigma_z=1/2*np.array([[1,0],[0,-1]])

    # Three contribution referred to three members in the sum of the L-qubits H are instantiated.
    H1 = np.zeros([2**L,2**L,]) 
    H2 = np.zeros([2**L,2**L,]) 
    H3 = np.zeros([2**L,2**L,]) 

    # Create the hamiltonian according to the number of qubits.
    if L == 1:
        H = -field*sigma_x - g*sigma_z
        
    else:
        # Spins nearest-neighbours interaction term.
        for i in range(1,L+1):   
            if i==1 or i==L:
                tempH1 =  copy.deepcopy(sigma_z)
            else:
                tempH1 = np.identity(2)
            for j in range(2,L+1):
                if j!=i-1 and ((j == i) or (j == i+1)):
                    tempH1 = np.kron(tempH1, sigma_z)
                else:
                    tempH1 = np.kron(tempH1,np.identity(2))
            H1 += tempH1  

        for j in range(L):
            # Static magnetic field and control magnetic field interaction terms.
            H3_temp = np.kron(sigma_x, np.identity(2**(L-j-1)))
            H3+=np.kron(np.identity(2**(j)), H3_temp)

            H2_temp = np.kron(sigma_z, np.identity(2**(L-j-1)))
            H2+=np.kron(np.identity(2**(j)), H2_temp)
            H = -(H1 + H2 + field*H3)

    #Compute and assign spectral quantities.
    eigval, eigvect = LA.eigh(H)
    spectral_dict = {"H":H ,"eigval":eigval , "eigvect":eigvect}
    return spectral_dict



class quantum_model:

    def __init__(self, qstart, qtarget, dt, L, g, h_list, history=True):

        self.qstart=qstart
        self.qtarget=qtarget
        self.dt=dt
        self.history=history

        self.L = L
        self.g = g
        self.h_list=h_list

        # Given self.h_list computes spectral quantities for each field value.
        self._init_hamiltonian() 
        self.reset()
        
    
    def reset(self):
        # Set current quantum state at qstart.
        self.qcurrent=self.qstart

        # Create list of visited states if history is true.
        self.qstates_history=[]
        if self.history:
            self.qstates_history.append(self.qcurrent)

        self.fidelity=None



    def _init_hamiltonian(self):
        ''' Create dictionary of dictionaries. 
        H_spectral_dict[field] contains a dictionary whose keys are "eigval", "eigvect" and "H" containing, infact, eigevalues, eigvectors of the
        hamiltonian H with that field value.'''

        self.H_spectral_dict = {field : compute_H_and_LA(self.L, self.g, field) for field in self.h_list}

    # Profile the bottlenecks in evolve function. 
    # @profile(sort_args=['name'], print_args=[25])

    def evolve(self, field, check_norm=True):
        ''' Given the value of the control field and considered the associated hamiltonian in H_spectral_dict, the self.qcurrent attribute 
        is evolved for that H and the dt.'''

        eigvect = copy.deepcopy(self.H_spectral_dict[field]["eigvect"])
        eigval  = copy.deepcopy(self.H_spectral_dict[field]["eigval"])

        # Compute a vector with entries coefficients for linear combination of eigenstates.
        c = np.dot(np.conj(eigvect.transpose()), self.qcurrent)*np.exp((-1j*eigval*self.dt))
        # Element-wise matrix multiplication, i.e. return a vector with at first element the product of the firsts elements, at the second one
        # of the second ones and so on and so forth. 
        eigvect = c*eigvect
        # Compute the sum of the 2^{L} elements in order to get the evolved state. 
        self.qcurrent = eigvect.sum(axis=1)

        # Norm checking for the sake of debugging with adequate tolerance. 
        if check_norm and (np.abs(1 - compute_fidelity_ext(self.qcurrent,self.qcurrent)) > 1e-9):
            print("Warning ---> Norm is not conserved.")

        if self.history:
            self.qstates_history.append(self.qcurrent)


    def compute_fidelity(self):
        ''' The function compute the fidelity for the two pure quantum states self.qtarget and self.qcurrent.'''
        self.fidelity = np.abs(np.vdot(self.qtarget, self.qcurrent) )**2
        return np.copy(self.fidelity)


    def evolve_from_protocol(self, protocol, make_gif=None):
        ''' The function for each value of the magnetic field h in the protocol compute the entire evolution 
        of the state after the application of the entire protocol.'''
        history_bool = self.history
        if not self.history:
            self.history=True

        for h in protocol:
            self.evolve(h)

        self.history=history_bool

        return np.copy(self.qstates_history)

def compute_fidelity_ext(qtarget, qcurrent):
    ''' The function external to the quantum mondel class and for this reason repeated 
    compute the fidelity for the two input pure quantum states qtarget and qcurrent.'''
    fidelity=np.abs(np.vdot(qtarget, qcurrent))**2
    return fidelity

def ground_state(L, field, g=1):
    ''' Given the dimension of the system L and the value of the control magnetic field, field, and the static one, g, the function
    return the eigenvector referred to the smallest eigenvalue of the hamiltonian, i.e. its ground state'''
    states = compute_H_and_LA(L,g,field)
    gstate = states["eigvect"][:,0]
    if (np.abs(1 - compute_fidelity_ext(gstate,gstate)) > 1e-9):
        print("Warning ---> Norm is not conserved")
        print(compute_fidelity_ext(gstate,gstate))
    return gstate

# Simple main of tasting with the cration of a gif for the sake of visualization.     
if __name__ == "__main__":

    import numpy as np
    from gif import create_gif

    qstart = np.array([-1/2 - (np.sqrt(5))/2 ,1], dtype=complex)
    qtarget = np.array([+1/2 + (np.sqrt(5))/2 ,1], dtype=complex)
    qstart=qstart/np.sqrt(np.vdot(qstart,qstart))
    qtarget=qtarget/np.sqrt(np.vdot(qtarget,qtarget))
    t_max=3
    n_steps=100
    dt = t_max/n_steps

    model = quantum_model(qstart, qtarget, dt, L=1, g=1, h_list=[-4, 0, 4], history=False)
    protocol=[0]*100
    states=model.evolve_from_protocol(protocol)
    fname = 'test.gif'
    create_gif(states, qstart, qtarget, fname)


# %%
