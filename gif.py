
#plotting on Bloch sphere
import matplotlib as mpl
from pylab import *
from qutip import *
from matplotlib import cm
import imageio

def qutip_qstate(coefs):
    up = basis(2,0)
    down = basis(2,1)
    return coefs[0]*up + coefs[1]*down


def create_gif(qstates, qstart, qtarget, name):
    ##---INPUTS
    # qstates: list of states as np.arrays
    # qstart, qtarget: respectively the target ans start state
    # name: name of the output gif

    b = Bloch()
    duration=0.005
    images=[]

    for qstate in qstates:
        b.clear()
        b.add_states(qutip_qstate(qstart))
        b.add_states(qutip_qstate(qtarget))
        b.add_states(qutip_qstate(qstate))
        filename='t.png'
        b.save(filename)
        images.append(imageio.imread(filename))

    imageio.mimsave(name, images, duration=duration)