'''
    Created on Oct 25th, 2020
    @authors: Alberto Chimenti, Clara Eminente and Matteo Guida.
    Purpose: (PYTHON3 IMPLEMENTATION)
        Stochastic Descent Program for the Work "Reinforcement Learning for Quantum Control".
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from SD import stochastic_descent,correlation
from Qmodel import compute_H_and_LA, compute_fidelity_ext, ground_state
import os
import argparse
from pathlib import Path
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

########################
## PARAMETERS ##########
########################
parser = argparse.ArgumentParser(prog = '\nStochastic Descent to find optimal protocol for a L-qubit system\n',
                            description = 'The program maps each protocol into a chain of classical spins and search for the configuration that minimizes the infidelity.\nThis procedure is carried out "iter_for_each_time" for each protocol duration and for a choosen time grid.')

parser.add_argument('--nsteps', type=int, nargs='?', default=100, help='Number of timesteps in the protocol')
parser.add_argument('--L', type=int, nargs='?', default=1, help='Number of qubits to consider')
parser.add_argument("--h", type=int, nargs="?", default=4, help='Control field value in bang-bang protocol')
parser.add_argument('--nflip', type=int, nargs='?', default=1, help='Number of flips at a time allowed')
parser.add_argument('--iter_for_each_time', type=int, nargs='?', default=20, help='Number of results to average for each fixed t.')

########################
########################
########################

if __name__ == "__main__":

    ### Parse input arguments
    args = parser.parse_args()
    
    h_list = [-args.h,args.h]

    print("------------------------------------------PARAMETERS for SD------------------------------------------")
    print("Number of qubits (L):", args.L)
    print("Magnetic fields(h):", args.h)
    print("Timesteps (n_steps):", args.nsteps)
    print("\n")
    print("\n")

    # Parameters for Fig. pag 2 PhysRevX.8.031086., i.e. calculation of fidelity and q(T).
    times_first_part=np.arange(0.1,1,0.1) 
    times_second_part=np.arange(1,4.1,0.1)
    times=np.concatenate([times_first_part,times_second_part])

    fidelity_for_txt = []
    print("------------------------------------------PARAMETERS for Plotting-------------------------------------")
    print("Timegrid:", times)
    print("Repetition at each timestep:", args.iter_for_each_time)
    print("\n")

    params_dict = {"L":args.L, "h":args.h, "timesteps":args.nsteps, "times":times, "iter_for_each_time": args.iter_for_each_time}
    params_df = pd.DataFrame.from_dict(params_dict, orient="index")


    # We set the ground states H at control fields hx = −2 and hx = 2 for the initial and target state.

    qstart = ground_state(args.L, -2)
    qtarget = ground_state(args.L, +2)

    start_fidelity = compute_fidelity_ext(qstart,qtarget)
    print("initial fidelity:",start_fidelity)

    # Save run parameters and date in custom named folder.
    custom_name_dir = "L"+str(args.L)+"_"+str(args.nflip)+"flip"
    Path(custom_name_dir).mkdir(exist_ok=True)
    Path(custom_name_dir+"/protocols").mkdir(exist_ok=True)

    params_df.to_csv(custom_name_dir+"/parameters.csv")

    intermediete_result = False

    # Iterate over the formed time-grid.
    for T in tqdm(times):
        temp_fid = []
        best_prot = []
        # For each time do iter_for_each_time for the sake of statistics. 
        for _ in range(args.iter_for_each_time):

            best_protocol, fidelity = stochastic_descent(qstart=qstart, qtarget=qtarget, L=args.L, T=T, nsteps=args.nsteps, nflip=args.nflip, 
                            field_list = h_list)

            # At fixed T we will have "iter_for_each_time" evaluations of fidelity.
            temp_fid.append(fidelity[-1])   
            # All iter_for_each_time best protocols are stored in this variable and saved when new run over T starts.
            best_prot.append(best_protocol)   
        # Fidelity evaluations are stored in the same "fidelity_fot_txt" variable that
        # Will have dimension len(times)*iter_for_each_time.
        fidelity_for_txt.append(temp_fid) 
                                                
        best_prot = np.array([best_prot])

        with open(custom_name_dir +'/protocols/testT'+str(round(T, 2))+'.npy', 'wb') as f:
            np.save(f,best_prot)
        f.close()

        if intermediete_result and T !=0: # If T = 0 q cannot be computed.
            data = np.load(custom_name_dir +'/protocols/testT'+str(round(T, 3))+'.npy')[0,:,:] # First dimension is redundant.
            print("Mean fidelity:", np.array(temp_fid).mean())
            print("Q value is:", correlation(data, args.h))
            print("\n")
            
    # Fidelity values are saved at the end.
    np.savetxt(custom_name_dir + '/fidelity_SD.txt', fidelity_for_txt, delimiter = ',',header="Matrix with as entries the values of fidelity dimension times x iterations")
    times=np.insert(times,0,0)



    # PLOTs. 
    q=[]
    for T in times[1:]:
        data = np.load(custom_name_dir +"/protocols/testT"+str(round(T, 2))+".npy")[0,:,:] #first dimension is redundant 
        q.append(correlation(data,args.h))

    times=np.concatenate([times_first_part,times_second_part])
    times=np.insert(times,0,0)
    loaded_fidelity = pd.read_csv(custom_name_dir +'/fidelity_SD.txt', skiprows=1,header=None)
    mean_fidelities = loaded_fidelity.mean(axis=1).values
    mean_fidelities=np.insert(mean_fidelities,0,start_fidelity)
    std_fidelities = loaded_fidelity.std(axis=1).values
    std_fidelities=np.insert(std_fidelities,0,0)

    q.insert(0,0)
    fig, ax = plt.subplots(figsize=(10,7))
    # Plot Fidelity values.
    ax.errorbar(times,mean_fidelities, yerr=std_fidelities, color="r")
    ax.scatter(times,mean_fidelities,color='r',label="Fidelity")
    # Plot Q values.
    ax.plot(times, q, marker="o", color="goldenrod", markersize=7, label="q(T)")
    ax.vlines(2.4,-0.05,1.05, color="b", linestyle="-.")
    ax.set_ylim(-0.05,1.05)
    ax.set_title(r"Phase Diagram L = "+str(args.L), fontsize=24)
    ax.set_xlabel("T [a.u.]", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=18, loc="center right",bbox_to_anchor=(1,0.3), shadow=True)
    ax.grid(linewidth=0.3)
    fig.tight_layout()
    fig.savefig(custom_name_dir+"/"+custom_name_dir+".pdf")
    plt.show()

