#%%
if __name__ == "__main__":

    from Qmodel import quantum_model, ground_state
    from QctRL import Agent
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    from gif import create_gif

    i = 0. + 1.j


    out_dir = Path("test")
    out_dir.mkdir(parents=True, exist_ok=True)

    ####### MODEL INIT #######
    ########## L=1 ###########
    # Define target and starting state
    qstart = np.array([-1/2 - (np.sqrt(5))/2 ,1], dtype=complex)
    qtarget = np.array([+1/2 + (np.sqrt(5))/2 ,1], dtype=complex)
    qstart=qstart/np.sqrt(np.vdot(qstart,qstart))
    qtarget=qtarget/np.sqrt(np.vdot(qtarget,qtarget))

    t_max=4
    n_steps=100
    dt = t_max/n_steps

    L=4
    qstart = ground_state(L, -2)
    qtarget = ground_state(L, +2)

    g=1
    all_actions = [-4, 0, 4]
    starting_action = 0
    episodes = 20001
    replay_freq=50
    replay_episodes=40

    model = quantum_model(qstart, qtarget, dt, L, g, all_actions)

    agent_init ={
        # Agent's initialization
        'nsteps' : n_steps,                  # number of training episodes
        'nactions' : len(all_actions),  
        'discount' : 1,                            # exponential discount factor
        'softmax' : True,
        'sarsa' : False
    }

    print("\nStarting with the following parameters:")
    print("---> T=", t_max)
    print("---> dt=", dt)
    print("---> N_states=", agent_init['nsteps']*agent_init['nactions'])
    print("---> Actions=", all_actions)

    # alpha value
    a=0.9
    eta=0.89
    alpha = np.linspace(a, eta, episodes)
    
    # initialize the agent
    learner = Agent(agent_init)
    learner._init_evironment(model, starting_action, all_actions)
    # train
    rewards, avg_rewards, epsilons = learner.train_agent(starting_action, episodes, alpha, replay_freq, replay_episodes, verbose=False)

    #### VARIOUS VISUALIZATION TASKS ####
    print("Best protocol Reward: {}".format(learner.best_reward))

    # plot reward results
    total_episodes=episodes+np.floor(episodes/replay_freq)*replay_episodes

    fname = 'train_result'+'_'+str(a)+'_'+str(eta)+'_'+str(t_max)+'.png'
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
 
    fname = 'protocol'+str(t_max)+'-'+str(dt)+'.gif'
    fname = out_dir / fname
    create_gif(learner.best_path, qstart, qtarget, fname)

#%%
    fig = plt.figure(figsize=(10,6))
    plt.plot(np.arange(len(learner.best_protocol)), learner.best_protocol)
    plt.show()