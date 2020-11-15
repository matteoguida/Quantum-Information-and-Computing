#%%
'''
    Created on Nov 9th, 2020
    @author: Alberto Chimenti, Clara Eminente and Matteo Guida
    Purpose: (PYTHON3 IMPLEMENTATION)
        Training wrapper script for QctRL.py agent
'''

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
#import sys

from Qmodel import quantum_model, ground_state
from QctRL import Agent
from gif import create_gif

########################
## PARAMETERS ##########
########################
parser = argparse.ArgumentParser(prog = '\nTrain RL Agent to find optimal protocol for a L-qubit system\n',
                            description = 'The program runs training of the RL Agent over the quantum system and outputs useful visualization plot of the performace.\nIf L=1 the program can produce a gif of the protocol representation on the Bloch sphere.')

parser.add_argument('--t_max', type=float, nargs='?', default=2.5, help='Total protocol time')
parser.add_argument('--nsteps', type=int, nargs='?', default=100, help='Number of timesteps in the protocol')
parser.add_argument('--L', type=int, nargs='?', default=1, help='Number of qubits to consider')
parser.add_argument('--g', type=int, nargs='?', default=1, help='Static field value')
parser.add_argument("--actions", type=int, nargs="+", default=[-4, 4], help='List of possible magnetic field values')
parser.add_argument("--starting_action", type=int, nargs="?", default=0, help='Starting action index')
parser.add_argument('--episodes', type=int, nargs='?', default=20001, help='Total number of episodes')
parser.add_argument('--replay_freq', type=int, nargs='?', default=50, help='Number of episodes to run between each replay session')
parser.add_argument('--replay_episodes', type=int, nargs='?', default=40, help='Number of replay episodes')
parser.add_argument('--out_dir', type=str, nargs='?', default='results', help='Output directory')
parser.add_argument('--gif', type=bool, nargs='?', default=False, help='Set equal to True if given L=1 a .gif animation of the protocol on the Bloch sphere is desired.')

########################
########################
########################

if __name__ == "__main__":

    ### Parse input arguments
    args = parser.parse_args()

    print("Running training with parameters:\n", args)

    #sys.stdout = open("train_profile.txt", "w")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = args.t_max/args.nsteps

    ####### MODEL INIT #######
    # Define target and starting state
    qstart = ground_state(args.L, -2)
    qtarget = ground_state(args.L, +2)
    model = quantum_model(qstart, qtarget, dt, args.L, args.g, args.actions)

    # alpha value
    a=0.9
    eta=0.89
    alpha = np.linspace(a, eta, args.episodes)
    
    # initialize the agent
    learner = Agent(args.nsteps, len(args.actions))
    learner._init_evironment(model, args.starting_action, args.actions)
    # train
    rewards, avg_rewards, epsilons = learner.train_agent(args.starting_action, args.episodes, alpha, args.replay_freq, args.replay_episodes, verbose=False)

    #### VARIOUS VISUALIZATION TASKS ####
    print("Best protocol Reward: {}".format(learner.best_reward))
    #sys.stdout.close()

    # save protocol
    fname = 'train_result_'+str(args.L)+'_'+str(args.t_max)+'.txt'
    fname = out_dir / fname
    data = [rewards, avg_rewards, epsilons]
    np.savetxt(fname, data, delimiter = ',')

    # plot reward results
    total_episodes=args.episodes+np.floor(args.episodes/args.replay_freq)*args.replay_episodes

    fname = 'train_result_'+str(args.L)+'_'+str(args.t_max)+'.png'
    fname = out_dir / fname
    plt.close('all')
    fig = plt.figure(figsize=(10,6))
    plt.scatter(np.arange(0,total_episodes+1,1), rewards, marker = '.', alpha=0.8)
    plt.scatter(np.arange(0,total_episodes,1), epsilons[1:], marker = '.', alpha=0.3)
    plt.scatter(np.arange(0,total_episodes,1), avg_rewards[1:], marker = '.', alpha=0.8)
    plt.xlabel('Episode number', fontsize=14)
    plt.ylabel('Fidelity', fontsize=14)
    plt.savefig(fname)
    plt.show()
    plt.close(fig=fig)
 
    if args.gif==True and args.L==1:
        fname = 'protocol'+str(args.t_max)+'-'+str(dt)+'.gif'
        fname = out_dir / fname
        create_gif(learner.best_path, qstart, qtarget, fname)
