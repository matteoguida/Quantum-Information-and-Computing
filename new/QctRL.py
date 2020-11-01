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
    lmbda = 1
    max_reward = 1
    qtable = np.matrix([1])
    softmax = False
    sarsa = False
    reward_bool = False
    
    # initialize
    def __init__(self, init_dict, qtable=None): 
     #nsteps, nactions, discount=0.9, max_reward=1, softmax=False, sarsa=False, qtable=None):
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
                print("WARNING ----> Qtable size doesn't match given arguments \n [nstates*nactions, nactions]=", [nstates*nactions, nactions], "\n Given:", np.shape(qtable))

    def _init_qtable(self):
        # initialize Q table [The indexing will be index(t)*len(h)+index(h)]
        # NOPE ---> NB: Adding #naction supplementary states in order to update the last ones
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
    def select_action(self, state, epsilon):

        qval = self.qtable[state] #selects a row in qtable
        prob = []

        if (self.softmax):# and epsilon!=0:
            if epsilon==0: epsilon=1
            # use Softmax distribution
            prob = sp.softmax(qval / epsilon) #epsilon controls the "temperature" in the softmax

        else:
            # use epsilon-greedy decision policy
            max_idx = np.argwhere(qval == np.max(qval)).flatten() # useful to avoid biased behaviour when choosing among flat distribution
            if len(max_idx) == self.nactions: epsilon=0
            # assign equal value to all actions
            prob = np.ones(self.nactions) * epsilon / (self.nactions - len(max_idx))
            # the best action is taken with probability 1 - epsilon
            prob[max_idx] = (1 - epsilon) / len(max_idx) # here epsilon chooses how greedy the action is

        return np.random.choice(range(0, self.nactions), p = prob)
        

    # update function (Sarsa and Q-learning)
    def update(self, action, alpha, epsilon):
        
        # update trace
        self.trace[self.env.state.previous, self.env.state.action] = 1

        observed = - self.qtable[self.env.state.previous, self.env.state.action] + self.env.reward

        if self.reward_bool:
            # for last time step iteration
            self.qtable += alpha * observed * self.trace
            return # to be checked

        # calculate long-term reward with bootstrap method
        ###### BEHAVIOURAL POLICY #######
        # find the next action (greedy for Q-learning, epsilon-greedy for Sarsa)
        if (self.sarsa):
            next_action = self.select_action(self.env.state.current, epsilon)
        else:
            next_action = self.select_action(self.env.state.current, 0)
        #################################

        # "bellman error" associated with the behavioural policy
        observed += self.discount * self.qtable[self.env.state.current, next_action]
        
        # bootstrap update
        self.qtable += alpha * observed * self.trace
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


    def train_episode(self, starting_action, alpha_vec, epsilon_vec):
        # intialize environement
        self.env.reset(starting_action)
        self.env.model.reset()
        self._init_trace()
        self.protocol = []

        for step in range(self.nsteps):

            self.reward_bool = (step == self.nsteps -1) #decides whether to compute reward or not

            # decision policy
            action = self.select_action(self.env.state.current, epsilon[step])

            # move environement current ---> previous
            self.env.move(action, self.reward_bool)

            self.protocol.append(self.env.all_actions[self.env.state.action])# append action to protocol
            self.env.model.evolve(self.env.all_actions[self.env.state.action])

            #######################################
            ### SET QUANTUM STATE model EVOLUTION EITHER HERE OR INSIDE ENVIRONEMENT MOVE CALL (cleaner but too hidden)
            #######################################

            self.update(action, alpha[step], epsilon[step])

'''
def generate_protocol(agent, qstart, start, mag_field, dt, time_ev_func, fidelity_func, episode_length, check_norm=True):
    agent.sarsa = False
    env = Environment(start=start, mag_field=mag_field, history=True)
    ev_qstate = [qstart]
    for j in range(0, episode_length):
        # find indexed version of the state 
        state_index = env.state[0] * len(mag_field) + env.state[1]
        # choose an action
        action = agent.select_action(state_index, 0)
        # the agent moves in the environment
        qstate = env.move(action, ev_qstate[j], dt, time_ev_func)
        # check norm conservation
        if check_norm and (np.abs(1 - fidelity_func(qstate, qstate)) > 1e-13):
            print("Warning ---> Norm is not conserved")
        ev_qstate.append(qstate)
    final_protocol = [env.action_map[env.path[j,1]] for j in np.arange(len(env.path[:,1]))]
    return final_protocol, ev_qstate
'''


def get_time_grid(t_max, dt):
    span = np.arange(0, t_max, dt, dtype=float)
    tdict = {i : span[i] for i in range(len(span))}
    return tdict


############################################################################


if __name__ == "__main__":

    ########## Standard initialization
    
    from Qmodel import quantum_model
    from pathlib import Path
    import matplotlib.pyplot as plt
    from gif import create_gif

    i = 0. + 1.j


    out_dir = Path("test")
    out_dir.mkdir(parents=True, exist_ok=True)

    ####### MODEL INIT #######
    # Define target and starting state
    qtarget = np.array([-1/np.sqrt(4) - 1/np.sqrt(4)*i, 1/np.sqrt(2) + 0.j])
    qstart = np.array([+1/np.sqrt(4) + 1/np.sqrt(4)*i, 1/np.sqrt(2) + 0.j])
    dt = 0.05

    model = quantum_model(qstart, qtarget, dt)
    model._init_inthamiltonian(L=1)
    all_actions = [-4, 0, 4]
    starting_action = 0
    t_max = 2.4
    episodes = 5001

    time_map = get_time_grid(t_max, dt)

    agent_init ={
        # Agent's initialization
        'nsteps' : len(time_map),                  # number of training episodes
        'nactions' : len(all_actions),  
        'discount' : 1,                            # exponential discount factor
        'max_reward' : 1,
        'softmax' : False,
        'sarsa' : True
    }

    print("\nStarting with the following parameters:")
    print("---> T=", t_max)
    print("---> dt=", dt)
    print("---> N_states=", agent_init['nsteps']*agent_init['nactions'])
    print("---> Actions=", all_actions)

    a = 0.9
    # alpha value and epsilon
    alpha = np.ones(episodes) * a
    epsilon = np.linspace(0.8, 0.01, episodes) #epsilon gets smaller i.e. action become more greedy as episodes go on

    
    # initialize the agent
    learner = Agent(agent_init)
    learner._init_evironment(model, starting_action, all_actions)

    from tqdm import tqdm

    # Train agent
    rewards = []
    best_reward = -1
    for index in tqdm(range(episodes)):

        learner.train_episode(starting_action, alpha, epsilon) # early stopping to implement
        rewards.append(learner.env.reward)

        #### BEST REWARD/PROTOCOL UPDATE ####
        if best_reward < learner.env.reward:
            best_protocol = learner.protocol
            best_reward = learner.env.reward
            best_path = learner.env.model.qstates_history
            print('New best protocol {} with reward {}'.format(index, best_reward))

    #### VARIOUS VISUALIZATION TASKS ####
    print("Best protocol Reward: {}".format(best_reward))
    
    # plot reward results
    fname = 'train_result'+'_'+str(a)+'.png'
    fname = out_dir / fname
    plt.close('all')
    fig = plt.figure(figsize=(10,6))
    plt.scatter(range(episodes), rewards, marker = '.', alpha=0.8)
    plt.xlabel('Episode number', fontsize=14)
    plt.ylabel('Fidelity', fontsize=14)
    plt.savefig(fname)
    plt.show()
    plt.close(fig=fig)

    fname = 'protocol'+str(t_max)+'-'+str(dt)+'.gif'
    fname = out_dir / fname
    create_gif(best_path, qstart, qtarget, fname)

    
#%%           
print(learner.qtable.shape)              
for i in range(learner.qtable.shape[0]):
    #if np.any(learner.qtable[i]!=0):
    print(learner.qtable[i], i)
