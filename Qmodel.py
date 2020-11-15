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
    
    '''

    The function creates an Hamiltonian given the number of qubits (L) and the field along the x-axis (field). See section 1.1 in the report
    for the form of the Hamiltonians. The funcion also computes its eigenvalues and eigenvectors and stores them in a dictionary.
    
    INPUTS:
    L: integer > 0, number of qubits in the system 
    g: float, static field along z-axis
    field: float, control field along x-axis (control field)
    
    
    OUTPUTS:
    spectarl_dict: dictionary, contains the hamiltonian along with its eigenvalues and eigenvectors.
            spectral_dict["H"]: a 2^L x 2^L numpy array (the hamiltonian)
            spectral_dict["eigval"]: 2^L numpy array (dtype=float) (the eigevalues of H)
            spectral_dict["eigvect"]: 2^L x 2^L numpy array (its columns are the eigevectors of H)
            
    '''

    from numpy import linalg as LA
    #if L < 1 breaks
    try:
        if L <= 0:
            raise ValueError

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

    except  ValueError:
        print("WARNING: number of Qubits L must be positive and not 0")


class quantum_model:
    '''

    This class implements the basics of our quantum model. Its main features are the computation of tha hamiltonian and its diagonalization and the time
    evolution function.
    
    INITIALIZATION VARIABLES:
    qstart, qtarget, qcurrent: np.array(dtype=complex) of size 2^L, respectively the initial, target and current quantum states
    dt: float, discrete timestep
    L: integer >0, number of Qubits
    g: float, static field along z-axis
    h_list: list of float, list of all possible field values to precompute and store eigenvalues and eignvectors of the corresponding hamiltonians
    history: boolean, if True the each evolved quantum state is stored to recreate the path

    '''
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
        '''

        The function resets the quantum system setting qcurrent=qstart and deleting the history

        '''
        # Set current quantum state at qstart.
        self.qcurrent=self.qstart

        # Create list of visited states if history is true.
        self.qstates_history=[]
        if self.history:
            self.qstates_history.append(self.qcurrent)

        self.fidelity=None



    def _init_hamiltonian(self):
        ''' 

        Create dictionary of dictionaries. 
        H_spectral_dict[field] contains a dictionary whose keys are "eigval", "eigvect" and "H" containing, infact, eigevalues, eigvectors of the
        hamiltonian H with that field value.

        '''
        self.H_spectral_dict = {field : compute_H_and_LA(self.L, self.g, field) for field in self.h_list}

    # Profile the bottlenecks in evolve function. 
    #@profile(sort_args=['name'], print_args=[25])
    def evolve(self, field, check_norm=True):
        ''' 

        Given the value of the control field and considered the associated hamiltonian in H_spectral_dict, the self.qcurrent attribute 
        is evolved for that H and the dt
        
        INPUTS:
        field: float, instanteneous value of the control field h^x
        check_norm: boolean, if True conservation of the norm is checked after evolution.

        '''

        eigvect = copy.deepcopy(self.H_spectral_dict[field]["eigvect"])
        eigval = self.H_spectral_dict[field]["eigval"]

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
        ''' 

        The function computes the fidelity for the two pure quantum states self.qtarget and self.qcurrent

        '''
        self.fidelity = np.abs(np.vdot(self.qtarget, self.qcurrent) )**2
        return np.copy(self.fidelity)


    def evolve_from_protocol(self, protocol):
        ''' 

        The function for each value of the magnetic field h^x in the protocol computes the entire evolution 
        of the state after the application of the entire protocol.
        
        '''
        #if history was set to false it is necessary to reset it to true. history_bool keeps track of this change and is used to reset it as it was 
        #at the end of the evolution.
        history_bool = np.copy(self.history)
        if not self.history:
            self.history=True

        for h in protocol:
            self.evolve(h)

        self.history=history_bool
        return np.copy(self.qstates_history)

def compute_fidelity_ext(qtarget, qcurrent):
    ''' 

    The function is used to compute the fidelity between two pure quantum states outside the class quantum_model.
    
    INPUTS:
    qtarget, qcurrent: np.array(dtype=complex), respectively the target and current quantum states

    OUTPUTS:
    fidelity: float, number between 0 and 1, the fidelity between the two states

    '''
    fidelity=np.abs(np.vdot(qtarget, qcurrent))**2
    return fidelity

def ground_state(L, field, g=1):
    ''' 

    Given the dimension of the system L and the value of the control magnetic field, field, and the static one, g, the function
    returns the eigenvector referred to the smallest eigenvalue of the hamiltonian, i.e. its ground state.
    
    INPUTS:
    L: integer > 0, number of qubts in the system
    g: float, static field along z-axis
    field: float, control field along x-axis (control field)

    OUTPUTS:
    gstate: np.array(), size 2^L, the coefficients of the ground state.

    '''

    try:
        if L <=0:
            raise ValueError
        states = compute_H_and_LA(L,g,field)
        gstate = states["eigvect"][:,0]
        if (np.abs(1 - compute_fidelity_ext(gstate,gstate)) > 1e-9):
            print("Warning ---> Norm is not conserved")
            print(compute_fidelity_ext(gstate,gstate))
        return gstate
    except ValueError:
        print("WARNING: number of Qubits L must be positive and not 0")

        

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
