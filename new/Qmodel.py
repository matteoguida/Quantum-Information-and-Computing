#%%
import numpy as np

class hamiltonian_eig:

    def __init__(self, L, g, field):
        self.L=L
        self.field=field
        self.g=g
        self.H=None
        self.eigvect=None
        self.eigvals=None
        #this initializes H and its spectral quantities
        self.compute_H_and_LA()


    def compute_H_and_LA(self):
        from numpy import linalg as LA
        #pauli matrices
        sigma_x=1/2*np.array([[0,1],[1,0]], dtype=complex)
        #sigma_y=np.array([[0,-1j],[1j,0]], dtype=complex) # INUTILE
        sigma_z=1/2*np.array([[1,0],[0,-1]], dtype=complex)
        sigma_z_interaction= np.kron(sigma_z,sigma_z)

        #create the hamiltonian according to the number of qubits
        if self.L == 1:
            self.H = -self.field*sigma_x - self.g*sigma_z
        else:
            H1 = np.zeros((2**self.L,2**self.L))
            H2 = np.zeros((2**self.L,2**self.L))
            H3 = np.zeros((2**self.L,2**self.L))

            for j in range(self.L-1):   
                H1 = np.kron(np.identity(2**j),sigma_z_interaction)
                H1 = np.kron(H1,np.identity(2**(self.L-j-2)))

            for j in range(self.L):
                H3 = np.kron(np.identity(2**j),sigma_x)
                H3 = np.kron(H3,np.identity(2**(self.L-j-1)))
                H2 = np.kron(np.identity(2**j),sigma_z)
                H2 = np.kron(H2,np.identity(2**(self.L-j-1)))

            self.H = -(H1 + self.g*H2 + self.field*H3)
        #compute and assign spectral quantities
        self.eigval, self.eigvect = LA.eig(self.H)
    


class quantum_model:

    def __init__(self, qstart, qtarget, dt, history=True):

        self.qstart=qstart
        self.qtarget=qtarget
        self.dt=dt
        self.history=history

        self.reset()
        
    
    def reset(self):
        #set current quantum state at qstart
        self.qcurrent=self.qstart

        #create list of visited states if history is true
        self.qstates_history=[]
        if self.history:
            self.qstates_history.append(self.qcurrent)

        self.fidelity=None

        #create hamiltonian variable which will comprehen
        self.hamiltonian=None

    def _init_inthamiltonian(self, L, g=1):
        self.L = L
        self.g = g

    def evolve(self, field):
        self.hamiltonian = hamiltonian_eig(self.L, self.g, field)
        c_i = np.array([np.vdot(self.hamiltonian.eigvect[:,i],self.qcurrent) for i in range(len(self.qcurrent))]) 
        temp_psi = []
        for i in range(len(self.qcurrent)):
            temp_psi.append(c_i[i]*np.exp((-1j*self.hamiltonian.eigval[i]*self.dt))*self.hamiltonian.eigvect[:,i])
        self.qcurrent = np.array(temp_psi).sum(axis=0)
        if self.history:
            self.qstates_history.append(self.qcurrent)


    def compute_fidelity(self):
        self.fidelity=np.abs(np.vdot(self.qtarget, self.qcurrent))**2
        return np.copy(self.fidelity)


    '''IDEA DI UTILIZZO
    Inizializzo Qmodel passandogli qtarget,qstate e dt (! ad inizio training)

    quando devo evolvere chiamo Qmodel.evolve(L, field, g): la funzione crea una classe hamiltonian che a sua volte sia rimpie calcolando 
    H e le sue quantità spettrali. Lo stato current (Qmodel.qcurrent) viene aggiornato.

    quando devo calcolare la fidelity chiamo Qmodel.compute_fidelity() (! prima devo evolvere un'ultima volta) che restituisce appunto la fidelity

    La classe hamiltonian e la classe Qmodel sono seprate perché Qmodel tiene traccia degli stati e della loro evoluzione mentre la classe 
    hamiltonian si occupa di gestire i cambiamenti dell'hamiltoniana CHE VA CAMBIATA OGNI VOLTA CON H!

    DOMANDE:
    -ha senso tenersi un Qmodel.hamiltonian ? alla fine non ci faccio granché. Forse conviene tenerla come variabile passeggera dentro evolve
    e fine. A quel punto sarebbe:
    hamiltonian = hamiltonian_eig(L,field, g)
    e tutto il resto hamiltonian.eigvect, hamiltonian.eigval (non Qmodel.hamiltonian.eigvect, Qmodel.hamiltonian.eigval)

    -L e g così vanno passati sempre ma tenerli dentro Qmodel mi sembrava poco coerente (sono oggetti che riguardano l'hamiltoniana quindi boh.)
    
    -come faccio ad inizializzare una classe dentro una funzione di un'altra classe? (mi serve per hamiltonian)
    '''


#if __name__ == "__main__":
#
#    import numpy as np
#
#    qtarget = np.array([-1/np.sqrt(4) - 1/np.sqrt(4)*1.j, 1/np.sqrt(2) + 0.j])
#    qstart = np.array([+1/np.sqrt(4) + 1/np.sqrt(4)*1.j, 1/np.sqrt(2) + 0.j])
#
#    model = quantum_model(qstart, qtarget, dt=0.01, history=True)
#    model.evolve(L=1, field=0, g=1)
#
#    print(model.qcurrent)
#
#    print(model.compute_fidelity())
# %%
