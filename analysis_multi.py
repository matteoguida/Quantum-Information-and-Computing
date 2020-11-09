#%%
from QctRL import protocol_analysis
import numpy as np
from Qmodel import ground_state
from pathlib import Path
import matplotlib.pyplot as plt
from gif import create_gif

if __name__ == "__main__":

    ########## Standard initialization

    i = 0. + 1.j


    out_dir = Path("test")
    out_dir.mkdir(parents=True, exist_ok=True)

    ####### MODEL INIT #######
    # Define target and starting state
    L=4

    qstart = ground_state(L, -2)
    qtarget = ground_state(L, +2)

    n_steps=100
    times_first_part=np.arange(0,1,0.1)
    times_second_part=np.arange(1,4.1,0.1)
    times=np.concatenate([times_first_part,times_second_part])
    print(times)
    h_list=[-4,0,4]

    fidelities = protocol_analysis(qstart, qtarget, times, n_steps, h_list, L=L)
    fname = out_dir / "fidelity_RL_L_"+str(L)+".txt"
    np.savetxt(fname, fidelities, delimiter = ',')