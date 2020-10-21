
#plotting on Bloch sphere
import matplotlib as mpl
from pylab import *
from qutip import *
from matplotlib import cm
import imageio
from mpl_toolkits.mplot3d import Axes3D

def qutip_qstate(coefs):
    up = basis(2,0)
    down = basis(2,1)
    return coefs[0]*up + coefs[1]*down


def create_gif(qstates, qstart, qtarget, name):
    ##---INPUTS
    # qstates: list of states as np.arrays
    # qstart, qtarget: respectively the target ans start state
    # name: name of the output gif

    fig = figure()
    ax = Axes3D(fig,azim=-40,elev=30)
    b = Bloch(axes=ax)


    b = Bloch()
    duration=0.05 #framerate
    images=[]

    for (qstate,i) in zip(qstates, np.arange(0,len(qstates))):

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
        ax.set_title(str(i), fontsize=30)
        b.save(filename)
        images.append(imageio.imread(filename))

    imageio.mimsave(name, images, duration=duration)