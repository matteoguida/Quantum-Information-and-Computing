'''
    Created on Oct 16, 2020

    @author: Alberto Chimenti

    Purpose: (PYTHON3 IMPLEMENTATION)
        General purpose Q-learning approach for optimal quantum control protocols
'''

#%%
import numpy as np
import scipy.special as sp

class Environment:

    def __init__(self, start=[0,0], mag_field=[-4, 0, +4], history=False):
        self.history = history
        self.state = np.asarray(start)
        self.action_map = {i : mag_field[i] for i in range(len(mag_field))}
        if self.history == True:
            self.path = np.array([self.state])

    # the agent makes an action (0 is -bound, LAST is +bound)
    def move(self, action, qstate, dt, time_ev_func):
        '''
        Given an action (i.e. next value of the field) this function computes the time evolved quantum state
        changing the hamiltonian with the new value of the field and returns the new qstate along with the new
        agent state (new time index e and new field index).

        Inputs:
        #action: integer, action index
        #qstate: np.array(dtype=complex), vector containing coefficients of the quantum state
        #dt: float, timestep
        #time_ev_func: function, function used for time evolution

        Outputs:
        #self.state: np.array, new state according to taken action
                     self.state[0]=indexed time, self.state[1]=field index
        #qstate: np.array(dtype=complex), vector containing coefficients of the time evolved quantum state 
        '''
        field = self.action_map[action]
        qstate = time_ev_func(qstate, dt, field)#, euler=True)
        self.state = [self.state[0]+1, action]
        if self.history:
            self.path = np.append(self.path, [self.state], axis=0)
        return self.state, qstate

class Agent:

    n_states = 1
    n_actions = 1
    discount = 0.9
    max_reward = 1
    qtable = np.matrix([1])
    softmax = False
    sarsa = False
    
    # initialize
    def __init__(self, nstates, nactions, discount=0.9, max_reward=1, softmax=False, sarsa=False, qtable=None):
        self.nstates = nstates
        self.nactions = nactions
        self.discount = discount
        self.max_reward = max_reward
        self.softmax = softmax
        self.sarsa = sarsa
        # initialize Q table
        # The indexing will be index(t)*len(h)+index(h)
        self.qtable = np.ones([nstates*nactions, nactions], dtype = float) * max_reward / (discount) #"normalize" the QTable
        if qtable is not None:
            qtable = np.array(qtable)
            if np.shape(qtable)==[nstates*nactions, nactions]:
                self.qtable = qtable
            else: 
                print("WARNING ----> Qtable size doesn't match given arguments \n [nstates*nactions, nactions]=", [nstates*nactions, nactions], "\n Given:", np.shape(qtable))

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
        '''

        The function implements the action choice policies, softmax and epsilon-greedy (i.e. the action with higher QValue
        is chosen with probability 1-epsilon).

        Inputs:
        #state: np.array, current agent state after taking the first action
                self.state[0]=indexed time, self.state[1]=field index
        #epsilon: float, "greedynes parameter"

        Outputs:
        #index of next action according to selected policy 
        '''
        qval = self.qtable[state] #selects a row in qtable
        prob = []
        if (self.softmax):
            # use Softmax distribution
            if epsilon == 0: epsilon = 1
            prob = sp.softmax(qval / epsilon) #epsilon controls how steep the softmax is (how greedy the choice is)
        else:
            # assign equal value to all actions
            prob = np.ones(self.nactions) * epsilon / (self.nactions - 1)
            # the best action is taken with probability 1 - epsilon
            prob[np.argmax(qval)] = 1 - epsilon
        return np.random.choice(range(0, self.nactions), p = prob)
        
    # update function (Sarsa and Q-learning)
    def update(self, state, action, reward, next_state, alpha, epsilon):
        '''

        The function updates the QTable in the following way:
        Q(s,a) <---- Q(s,a)*(1-alpha) + alpha*(reward + discount * Q[next_s, next_a])

        If "sarsa" the next action is chosen using epsilon greedy policy
        If not, next_action maximizes the value of the QTable for next_state.


        Inputs:
        #state: integer, indexed version of current state
        #action: integer, index of the value of the new field
        #reward: float, reward
        #next_state: integer, indexed version of current state after being in "state" and taking the action indexed by "action"
        #alpha: float, learning rate
        #epsilon: float, "greedyness parameter" for action choice policy

        Outputs:
        #updated QTable 
        '''
        # find the next action (greedy for Q-learning, using the decision policy for Sarsa)
        next_action = self.select_action(next_state, 0)
        if (self.sarsa):
            next_action = self.select_action(next_state, epsilon)
        # calculate long-term reward with bootstrap method
        observed = reward + self.discount * self.qtable[next_state, next_action]
        # bootstrap update
        self.qtable[state, action] = self.qtable[state, action] * (1 - alpha) + observed * alpha

    # simple output directory selector
    def get_out_dir(self):
        if self.sarsa==True:
            name = 'sarsa'
        else:
            name = 'no_sarsa'
        if self.softmax==True:
            name = name + '_softmax'
        return name


def train_agent(agent, qtarget, qstart, start, mag_field, dt, time_ev_func, 
                fidelity_func, episodes, episode_length, epsilon, alpha, verbose=None, check_norm=True, make_gif=None):
    '''
    Training function. The function trains the agent for n=episodes episodes, each of whom lasts episode_length time steps of length dt.
    Fidelity_func is used as the reward function.

    !WARNING: as of 20.10.2020 the function also creates a gif with the evolution of the quantum state every 2000 episides

    Inputs:
    #agent: object of class Agent
    #qtarget: np.array(dtype=complex), vector containing coefficients of the TARGET quantum state
    #qstart: np.array(dtype=complex), vector containing coefficients of the STARTING quantum state
    #start: np.array, starting agent state 
            self.state[0]=indexed time, self.state[1]=field index
    #mag_field: list, list of values that the agent can be in
    #dt: float, timestep
    #time_ev_func: function, time evolution function
    #fidelity_func: function,  used to compute the rewards
    #episodes: integer, number of training episodes
    #episodes_length: float, length of a single training episode
    #epsilon: float, "greedyness parameter"
    #alpha: float, learning rate
    #verbose: integer, if not None, informations on the status of learning are provided every "Verbose" episodes
    #make_gif: integer, if not None, a gif with the path of the state is created every "make_gif" episodes
    #check_norm: boolean, if "True", check on norm conservatio  is done everytime the state is evolved

    Outputs:
    #env: evolved environment (state and qstate)
    #reward: list, list of the rewards obtained
    #paths: list, list of lists containing paths (saved every "verbose" episodes)
    '''
    from tqdm import tqdm
    from gif import create_gif

    rewards = []
    #-----CLARA------------------------
    #storing paths for plotting purposes
    paths = []
    #---------------------------------
    for index in tqdm(range(0, episodes)):
        # initialize environment
        env = Environment(start=start, mag_field=mag_field, history=True)
        state = env.state
        qstate = qstart
        reward = 0
        #-----CLARA------------------------
        #storing states for plotting purposes
        qstates = []
        #---------------------------------
        # run episode
        for _ in range(0, episode_length):
            # find indexed version of the state 
            state_index = state[0] * len(mag_field) + state[1]
            # choose an action
            action = agent.select_action(state_index, epsilon[index])
            # the agent moves in the environment
            state, qstate = env.move(action, qstate, dt, time_ev_func)
            # check norm conservation
            if check_norm and (np.abs(1 - fidelity_func(qstate, qstate)) > 1e-13):
                print("Warning ---> Norm is not conserved")
            # compute reward
            #if step == episode_length-1: # Uncomment if only the last reward has to be counted
            reward = fidelity_func(qtarget, qstate)
            # Q-learning update
            next_index = state[0] * len(mag_field) + state[1]
            agent.update(state_index, action, reward, next_index, alpha[index], epsilon[index])
        #-----CLARA------------------------------
        #storing states for plotting purposes
        qstates.append(qstate)

        if (make_gif is not None) and ((index) % make_gif == 0):
            create_gif(qstates, qstart, qtarget, "bloch_anim_"+str(index)+".gif")
        #----------------------------------------
        rewards.append(reward)
        # periodically save the agent
        if (verbose is not None) and ((index + 1) % verbose == 0):
            #agent_state = learner.extract_state()
            #name = 'agent'+'_'+str(a)+'.obj'
            #with open(out_dir / name, 'wb') as agent_file:
            #    dill.dump(agent_state, agent_file)
            paths.append(env.path) #stores every verbose paths
            print('\nEpisode ', index + 1, ': the agent has obtained fidelity eqal to', reward, '\nStarting from position ', qstart)
    return env, rewards, qstates


def get_time_grid(t_max, dt):
    span = np.arange(0, t_max, dt, dtype=float)
    tdict = {i : span[i] for i in range(len(span))}
    return tdict


############################################################################


if __name__ == "__main__":

    ########## Standard initialization
    from quantum_state import i, time_evolution, compute_fidelity
    from pathlib import Path
    import matplotlib.pyplot as plt

    #-----CLARA-------------------------------------------
    #custom module for visualization purpose
    #from gif import *
    #-----------------------------------------------------

    out_dir = Path("test")
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = 5000         # number of training episodes
    discount = 0.9          # exponential discount factor
    t_max = 0.1               # simulation time in seconds
    dt = 0.01               # timestep in seconds
    time_map = get_time_grid(t_max, dt)
    episode_length = len(time_map)         # maximum episode length
    mag_field = [-4, 0, +4]
    nstates = episode_length*len(mag_field) # total number of states
    nactions = len(mag_field)              # total number of possible actions

    # Define target and starting state
    qtarget = np.array([0. + 0.j,-1/np.sqrt(2)-1/np.sqrt(2)*i])
    qstart = np.array([+1/np.sqrt(2)+1/np.sqrt(2)*i,0. + 0.j])

    a = 0.9
    # alpha value and epsilon
    alpha = np.ones(episodes) * a
    epsilon = np.linspace(0.8, 0.001,episodes) #epsilon gets smaller i.e. action become more greedy as episodes go on
    # initialize the agent
    learner = Agent(nstates, nactions, discount, max_reward=1, softmax=True, sarsa=True)
    # perform the training
    start = [0,0]
    env, rewards, _ = train_agent(learner, qtarget, qstart, start, mag_field,
                    dt, time_evolution, compute_fidelity, episodes, episode_length, 
                    epsilon, alpha)

    # plot result
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
