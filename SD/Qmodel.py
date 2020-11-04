#%%
import numpy as np


def compute_H_and_LA(L, g, field):
    from numpy import linalg as LA
    #pauli matrices
    sigma_x=1/2*np.array([[0,1],[1,0]], dtype=complex)
    #sigma_y=np.array([[0,-1j],[1j,0]], dtype=complex) # INUTILE
    sigma_z=1/2*np.array([[1,0],[0,-1]], dtype=complex)
    sigma_z_interaction= np.kron(sigma_z,sigma_z)

    #create the hamiltonian according to the number of qubits
    if L == 1:
        H = -field*sigma_x - g*sigma_z
        
    else:
        H1 = np.zeros((2**L,2**L), dtype="complex128")
        H2 = np.zeros((2**L,2**L), dtype="complex128")
        H3 = np.zeros((2**L,2**L), dtype="complex128")

        for j in range(L-1):   
            
            H1_temp = np.kron(sigma_z_interaction, np.identity(2**(L-j-2)))
            H1+=np.kron(np.identity(2**(j)), H1_temp)

        for j in range(L):

            H3_temp = np.kron(sigma_x, np.identity(2**(L-j-1)))
            H3+=np.kron(np.identity(2**(j)), H3_temp)

            H2_temp = np.kron(sigma_z, np.identity(2**(L-j-1)))
            H2+=np.kron(np.identity(2**(j)), H2_temp)

        H = -(H1 + g*H2 + field*H3)
        print(H)
    #compute and assign spectral quantities
    eigval, eigvect = LA.eig(H)
    spectral_dict = {"eigval":eigval , "eigvect":eigvect}
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

        self._init_hamiltonian() #given self.h_list computes spectral quantities for each field value
        self.reset()
        
    
    def reset(self):
        #set current quantum state at qstart
        self.qcurrent=self.qstart

        #create list of visited states if history is true
        self.qstates_history=[]
        if self.history:
            self.qstates_history.append(self.qcurrent)

        self.fidelity=None



    def _init_hamiltonian(self):
        '''This is a dictionary of dictionaries. 
        H_spectral_dict[field] contains a dictionary whose labels are "eigval" and "eigvect" containing, infact, eigevalues and eigvectors of the
        hamiltonian with that field value.'''

        self.H_spectral_dict = {field : compute_H_and_LA(self.L, self.g, field) for field in self.h_list}


    def evolve(self, field):
        eigvect = self.H_spectral_dict[field]["eigvect"]
        eigval = self.H_spectral_dict[field]["eigval"]                                                  
        c_i = np.array([np.vdot(eigvect[:,i],self.qcurrent) for i in range(len(self.qcurrent))]) 
        temp_psi = []
        for i in range(len(self.qcurrent)):
            temp_psi.append(c_i[i]*np.exp((-1j*eigval[i]*self.dt))*eigvect[:,i])
        self.qcurrent = np.array(temp_psi).sum(axis=0)
        if self.history:
            self.qstates_history.append(self.qcurrent)


    def compute_fidelity(self):
        self.fidelity=np.abs(np.vdot(self.qtarget, self.qcurrent))**2
        return np.copy(self.fidelity)


    def evolve_from_protocol(self, protocol, make_gif=None):
        
        history_bool = self.history
        if not self.history:
            self.history=True

        for h in protocol:
            self.evolve(h)

        self.history=history_bool

        return np.copy(self.qstates_history)

    
'''if __name__ == "__main__":

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
    create_gif(states, qstart, qtarget, fname)'''


# %%
a = compute_H_and_LA(2,1,4)


sigma_z=1/2*np.array([[1,0],[0,-1]], dtype=complex)
sigma_x=1/2*np.array([[0,1],[1,0]], dtype=complex)

H3=np.kron(sigma_x,np.identity(2))
H6=np.kron(np.identity(2),sigma_x)
#print(H3+H6)

H2=np.kron(sigma_z,np.identity(2))
H5=np.kron(np.identity(2),sigma_z)
#print(H2+H5)
# %%
