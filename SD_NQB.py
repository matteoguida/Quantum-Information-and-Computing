'''
    Created on Oct 25th, 2020
    @authors: Alberto Chimenti, Clara Eminente and Matteo Guida.
    Purpose: (PYTHON3 IMPLEMENTATION)
        Stochastic Descent Program for the Work "Reinforcement Learning for Quantum Control".
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm,tnrange
from SD import stochastic_descent,correlation
from Qmodel import compute_H_and_LA, compute_fidelity_ext, ground_state
import os
from pathlib import Path
import warnings
import pandas as pd
warnings.filterwarnings('ignore')



#SD parameters.
L = 3 # Number of qubits of the quantum system.
h = 4 # Control field value in bang-bang protocol. 
h_list = [-h,h]
nsteps = 100 # Lenght of the protocol.
exp_decay = False
metropolis = False

print("------------------------------------------PARAMETERS for SD------------------------------------------")
print("Number of qubits (L):", L)
print("Magnetic fields(h):", h)
print("Timesteps (n_steps):", nsteps)
print("\n")
print("\n")

# Parameters for Fig. pag 2 PhysRevX.8.031086., i.e. calculation of fidelity and q(T).
times_first_part=np.arange(0.1,1,0.1) 
times_second_part=np.arange(1,4.1,0.1)
times=np.concatenate([times_first_part,times_second_part])

nflip=1 # Number of flips in SD algorithm.
iter_for_each_time = 20 # Number of iteration for each fixed t.

fidelity_for_txt = []
print("------------------------------------------PARAMETERS for Plotting-------------------------------------")
print("Timegrid:", times)
print("Repetition at each timestep:", iter_for_each_time)
print("\n")

params_dict = {"L":L, "h":h, "timesteps":nsteps, "exp_decay":exp_decay, "metropolis":metropolis, "times":times, "iter_for_each_time": iter_for_each_time}
params_df = pd.DataFrame.from_dict(params_dict, orient="index")


# We set the ground states H at control fields hx = −2 and hx = 2 for the initial and target state.

qstart = ground_state(L, -2)
qtarget = ground_state(L, +2)
    
start_fidelity = compute_fidelity_ext(qstart,qtarget)
print("initial fidelity:",start_fidelity)

# Save run parameters and date in custom named folder.
custom_name_dir = "L"+str(L)+"_"+str(nflip)+"flip"
Path(custom_name_dir).mkdir(exist_ok=True)
Path(custom_name_dir+"/protocols").mkdir(exist_ok=True)

params_df.to_csv(custom_name_dir+"/parameters.csv")

intermediete_result = True

# Iterate over the formed time-grid.
for T in tqdm(times):
    temp_fid = []
    best_prot = []
    # For each time do iter_for_each_time for the sake of statistics. 
    for _ in tnrange(iter_for_each_time):

        best_protocol, fidelity = stochastic_descent(qstart=qstart, qtarget=qtarget, L=L, T=T, nsteps=nsteps, nflip=nflip, 
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
        print("Q value is:", correlation(data, h))
        print("\n")
        
# Fidelity values are saved at the end.
np.savetxt(custom_name_dir + '/fidelity_SD.txt', fidelity_for_txt, delimiter = ',',header="Matrix with as entries the values of fidelity dimension times x iterations")
times=np.insert(times,0,0)



# PLOTs. 
q=[]
for T in times[1:]:
    data = np.load(custom_name_dir +"/protocols/testT"+str(round(T, 2))+".npy")[0,:,:] #first dimension is redundant 
    q.append(correlation(data,h))

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
ax.vlines(0.5,-0.05,1.05, color="b", linestyle="-.")
ax.vlines(2.4,-0.05,1.05, color="b", linestyle="-.")
ax.set_ylim(-0.05,1.05)
ax.set_title(r"Phase Diagram L = "+str(L), fontsize=24)
ax.set_xlabel("t [a.u.]", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(fontsize=18, loc="center right",bbox_to_anchor=(1,0.3), shadow=True)
ax.grid(linewidth=0.3)
fig.tight_layout()
fig.savefig(custom_name_dir+"/"+custom_name_dir+".pdf")
plt.show()

