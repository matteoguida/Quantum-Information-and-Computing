'''
    Created on Oct 31st, 2020
    @author: Alberto Chimenti
    Purpose: (PYTHON3 IMPLEMENTATION)
        General purpose Q-learning approach for optimal quantum control protocols
'''

#%%
import numpy as np
from environment import Environment
import scipy.special as sp


class Agent:

    n_states = 1
    n_actions = 1
    discount = 0.9
    lmbda = 0.8
    max_reward = 1
    qtable = np.matrix([1])
    softmax = False
    sarsa = False
    reward_bool = False
    
    # initialize
    def __init__(self, init_dict, qtable=None): 
        self.nsteps = init_dict['nsteps']
        self.nactions = init_dict['nactions']
        self.nstates = self.nsteps*self.nactions

        self.discount = init_dict['discount']
        #self.max_reward = max_reward

        self.softmax = init_dict['softmax']
        self.sarsa = init_dict['sarsa']

        self._init_qtable()
        self._init_trace()
        if qtable is not None:
            qtable = np.array(qtable)
            if np.shape(qtable)==[self.nstates, self.nactions]:
                self.qtable = qtable
            else: 
                print("WARNING ----> Qtable size doesn't match given arguments \n [nstates*nactions, nactions]=", [self.nstates*self.nactions, self.nactions], "\n Given:", np.shape(qtable))

    def _init_qtable(self):
        # initialize Q table [The indexing will be index(t)*len(h)+index(h)]
        self.qtable = np.zeros([self.nstates, self.nactions], dtype = float)

    def _init_trace(self):
        self.trace = np.zeros([self.nstates, self.nactions], dtype = float)

    def _init_evironment(self, model, starting_action, all_actions, history=True):
        self.env = Environment(model, starting_action, all_actions, history)

    def extract_state(self):
        state_dict = {
            'nstates' : self.nstates,
            'nactions' : self.nactions,
            'discount' : self.discount,
            'max_reward' : self.max_reward,
            'softmax' : self.softmax,
            'sarsa' : self.sarsa,
            'qtable' : self.qtable
            }
        return state_dict

    # action policy: implements epsilon greedy and softmax
    def select_action(self, state, epsilon, greedy=False, replay=False):

        if replay: #action is that of the best protocol at that time step
            action = self.best_protocol[self.env.time_step] #action is not indexed, is the actual value
            indA = self.env.action_map_dict[action] #return indexed version

        else:
            qval = self.qtable[state] #selects a row in qtable
            prob = []

            max_idx = np.argwhere(qval == np.max(qval)).flatten() # useful to avoid biased behaviour when choosing among flat distribution
            greedy_action = np.random.choice(max_idx)

            if not greedy:
                if (self.softmax):
                    if epsilon==0: epsilon=1
                    # use Softmax distribution
                    prob = sp.softmax(qval / epsilon) #epsilon controls the "temperature" in the softmax
                    indA = np.random.choice(range(0, self.nactions), p = prob)
                
                else:
                    # use epsilon-greedy decision policy
                    if len(max_idx) == self.nactions: epsilon=0
                    # assign equal value to all actions
                    prob = np.ones(self.nactions) * epsilon / (self.nactions - len(max_idx))
                    # the best action is taken with probability 1 - epsilon
                    prob[max_idx] = (1 - epsilon) / len(max_idx) # here epsilon chooses how greedy the action is
                    indA = np.random.choice(range(0, self.nactions), p = prob)

                if indA!=greedy_action:
                    self._init_trace()
            else:
                indA = greedy_action
            
        
        return indA
        

    # update function (Sarsa and Q-learning)
    def update(self, action, alpha, epsilon):
        
        # update trace
        self.trace[self.env.state.previous, self.env.state.action] = alpha

        observed = - self.qtable[self.env.state.previous, self.env.state.action] + self.env.reward

        if self.reward_bool:
            # for last time step iteration
            self.qtable += alpha * observed * self.trace
            return

        # calculate long-term reward with bootstrap method
        ###### BEHAVIOURAL POLICY #######
        # find the next action (greedy for Q-learning, epsilon-greedy for Sarsa)
        #if (self.sarsa):
        #    next_action = self.select_action(self.env.state.current, epsilon)
        #else:
        next_action = self.select_action(self.env.state.current, 0, greedy=True)
        #################################

        # "bellman error" associated with the behavioural policy
        observed += self.discount * self.qtable[self.env.state.current, next_action]
        
        # bootstrap update
        self.qtable += observed * self.trace
        self.trace *= (self.discount * self.lmbda)


    # simple output directory selector
    def get_out_dir(self):
        if self.sarsa==True:
            name = 'sarsa'
        else:
            name = ''
        if self.softmax==True:
            name = name + '_softmax'
        return name


    def train_episode(self, starting_action, alpha, epsilon, replay=False):
        # intialize environement
        self.env.reset(starting_action)
        self.env.model.reset()
        self._init_trace()
        self.protocol = []

        for step in range(self.nsteps):

            #print(step)

            self.reward_bool = (step == self.nsteps - 1) #decides whether to compute reward or not

            # decision policy
            action = self.select_action(self.env.state.current, epsilon, replay=replay) #greedy=False by default

            # evolve quantum model
            self.env.model.evolve(self.env.all_actions[action])

            # move environement current ---> previous
            self.env.move(action, self.reward_bool)

            # append action to protocol
            self.protocol.append(self.env.all_actions[self.env.state.action])

            # update agent's Q-table
            self.update(action, alpha, epsilon)
            

    def train_agent(self, starting_action, episodes, alpha_vec, replay_freq, replay_episodes, verbose=False, epsilon_i=1, epsilon_f=0, test=10):
        from tqdm import tqdm

        # Train agent
        rewards = []
        self.best_reward = -1

        #############################
        self.epsilon_f = epsilon_f
        self.epsilon_i = epsilon_i
        epsilons = [self.epsilon_i]
        epsilon = self.epsilon_i
        self.counter = 0
        mavg_rewards = [0]
        avg_reward = 0
        self.avg_reward = avg_reward
        #############################

        for index in tqdm(range(episodes)):

            self.train_episode(starting_action, alpha_vec[index], epsilon, replay=False)
            rewards.append(self.env.reward)
            mavg_rewards.append(((mavg_rewards[-1]*index) + self.env.reward)/(index+1))

            #############################
            #avg_reward += self.env.reward
            if index%20==0:
                epsilon = self.update_greedyness(episodes, index, epsilon, mavg_rewards[-1])
                #avg_reward = 0
            epsilons.append(epsilon)
            #############################

            #### BEST REWARD/PROTOCOL UPDATE ####
            if self.best_reward < self.env.reward:
                self.best_protocol = self.protocol
                self.best_reward = self.env.reward
                self.best_path = self.env.model.qstates_history
                if verbose:
                    print('\nNew best protocol {} with reward {}'.format(index, self.best_reward))

            # Earlystopping


            if index%replay_freq==0 and index!=0:
                for _ in range(replay_episodes):
                    self.train_episode(starting_action, alpha_vec[index], epsilon, replay=True)
                    rewards.append(self.env.reward)
                    mavg_rewards.append(((mavg_rewards[-1]*index) + self.env.reward)/(index+1))
                    #############################
                    epsilons.append(epsilon)
                    #############################
                    #self.best_reward=self.env.reward ##NON SICURISSIMA DI QUESTO (tipo, se viene più bassa?)
                    if verbose:
                        print("\n...Running greedy epidosdes...")
                        #print('New best reward {}'.format(self.best_reward))

        # Test learning
        _, reward = self.generate_protocol(starting_action)
        rewards.append(reward)

        if test is not None:
            print("----> Testing convergence...")
            test_sum = 0
            for _ in range(test):
                _, reward = self.generate_protocol(starting_action)
                test_sum += reward
            error = np.abs(self.best_reward - test_sum/test)
            if error > 1e-3:
                print("!WARNING: The Q-table does not converge. Deviating {} from best protocol fidelity".format(error))
            else:
                print("Learning seems to be fine!")

        return rewards, mavg_rewards, epsilons


    def update_greedyness(self, episodes, episode, epsilon, avg_reward, max_steps=10, T=8):
        # max_steps: decides how many steps to tolerate and how much increment to consider for reward threshold
        # T: is the temperature factor in the exponential decay of the epsilon parameter
        if (avg_reward >= self.avg_reward) or (self.counter >= max_steps):
            self.avg_reward += (avg_reward-self.avg_reward)*(max_steps/100) # adds 10% of the increment
            epsilon = self.epsilon_f + (self.epsilon_i - self.epsilon_f)*np.exp(-T*episode/episodes)
            self.counter = 0
        else:
            self.counter += 1
        return epsilon


    def generate_protocol(self, starting_action, **kwargs):
        # intialize environement
        self.env.reset(starting_action)
        if 'qstart' in kwargs: 
            self.env.model.qstart = kwargs.get('qstart')
        self.env.model.reset()
        self.protocol = []

        for step in range(self.nsteps):

            self.reward_bool = (step == self.nsteps - 1) #decides whether to compute reward or not

            # decision policy
            action = self.select_action(self.env.state.current, 0, greedy=True) #greedy=False by default

            # evolve quantum model
            self.env.model.evolve(self.env.all_actions[action])

            # move environement current ---> previous
            self.env.move(action, self.reward_bool)

            # append action to protocol
            self.protocol.append(self.env.all_actions[self.env.state.action])

        return self.protocol, self.env.reward


def protocol_analysis(qstart, qtarget, t_max_vec, n_steps, all_actions, **kwargs):

    L=1
    g=1
    starting_action = 0
    episodes = 20001
    replay_freq=50
    replay_episodes=40

    if 'L' in kwargs:
        L = kwargs.get('L')
        print("Overwritten default L with:",L)
    if 'g' in kwargs:
        g = kwargs.get('g')
    if 'starting_action' in kwargs:
        starting_action = kwargs.get('starting_action')
    if 'episodes' in kwargs:
        episodes = kwargs.get('episodes')
    if 'replay_freq' in kwargs:
        replay_freq = kwargs.get('replay_freq')
    if 'replay_episodes' in kwargs:
        replay_episodes = kwargs.get('replay_episodes')

    # alpha value
    a=0.9; eta=0.89
    alpha = np.linspace(a, eta, episodes)

    agent_init ={
        # Agent's initialization
        'nsteps' : n_steps,                  # number of training episodes
        'nactions' : len(all_actions),  
        'discount' : 1,                            # exponential discount factor
        'softmax' : True,
        'sarsa' : False
    }

    fidelities = []
    for _, t_max in enumerate(t_max_vec):
        print("\n Running training for T={}".format(t_max))

        dt = t_max/n_steps
        model = quantum_model(qstart, qtarget, dt, L, g, all_actions)

        # initialize the agent
        learner = Agent(agent_init)
        learner._init_evironment(model, starting_action, all_actions)
        # train
        _ = learner.train_agent(starting_action, episodes, alpha, replay_freq, replay_episodes, verbose=False)
        print("Found protocol with fidelity:", learner.best_reward)
        fidelities.append([t_max, learner.best_reward])
        #### or ####
        #_, R = learner.generate_protocol(starting_action)
        #fidelities.append([t_max, R])
    
    return fidelities




############################################################################


if __name__ == "__main__":

    ########## Standard initialization
    
    from Qmodel import quantum_model, ground_state
    from pathlib import Path
    import matplotlib.pyplot as plt
    from gif import create_gif

    i = 0. + 1.j


    out_dir = Path("test")
    out_dir.mkdir(parents=True, exist_ok=True)

    ####### MODEL INIT #######
    ########## L=1 ###########
    # Define target and starting state
    #qstart = np.array([-1/2 - (np.sqrt(5))/2 ,1], dtype=complex)
    #qtarget = np.array([+1/2 + (np.sqrt(5))/2 ,1], dtype=complex)
    #qstart=qstart/np.sqrt(np.vdot(qstart,qstart))
    #qtarget=qtarget/np.sqrt(np.vdot(qtarget,qtarget))

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
    fname = out_dir / "fidelity_RL.txt"
    np.savetxt(fname, fidelities, delimiter = ',')

#%%
    plt.plot(times, fidelities, linesyle='-.')



#%%



    t_max=4
    n_steps=100
    dt = t_max/n_steps

    L=1
    g=1
    all_actions = [-4, 0, 4]
    starting_action = 0
    episodes = 20001
    replay_freq=50
    replay_episodes=40

    model = quantum_model(qstart, qtarget, dt, L, g, all_actions)

    #time_map = get_time_grid(t_max, dt)

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
print(learner.qtable.shape)              
for i in range(learner.qtable.shape[0]):
    #if np.any(learner.qtable[i]!=0):
    print(learner.qtable[i], i)
