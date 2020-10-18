'''
    Created on Oct 16, 2020

    @author: Alberto Chimenti

    Purpose: (PYTHON3 IMPLEMENTATION)
        General purpose Q-learning approach for optimal quantum control protocols
'''

#%%
import numpy as np
### Import time evolution library

class Environment:
    state = []
    goal = []

    def __init__(self, qtarget=[1,1], qstart=[0,0], start=[0,-2], mag_field=[-4, -2, 0, +2, +4], history=False):
        # Check coefficients
        if len(qtarget)!=len(qstart): print("Warning: target and init coeffitients number don't match! \nExiting..."); break
        ### Quantum state data (not sure whether to save it in the environement)
        self.qtarget = qtarget
        self.qstart = qstart
        self.qstate = qstart
        self.nqubit = len(qstart)/2
        ###
        self.history = history
        self.state = np.asarray(start)
        self.action_map = {i : mag_field[i] for i in range(len(mag_field))}
        if self.history == True:
            self.path = np.array([self.state])

    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action):#, qstate, reward_func):
        field = self.action_map[action]
        ########
        self.qstate = ### time_ev(self.qstate, self.nqubit, field)
        reward = ### hs_dist(self.qtarget, self.qstate)
        ########
        ################
        reward = ### reward_func(qstate, field)
        ################
        self.state = [self.state[0]+1, field]
        if self.history:
            self.path = np.append(self.path, [self.state], axis=0)
        return [self.state, reward]

class Agent:

    n_states = 1
    n_actions = 1
    discount = 0.9
    max_reward = 1
    qtable = np.matrix([1])
    softmax = False
    sarsa = False
    
    # initialize
    def __init__(self, n_states, n_actions, discount=0.9, max_reward=1, softmax=False, sarsa=False, qtable=None):
        self.nstates = n_states
        self.nactions = n_actions
        self.discount = discount
        self.max_reward = max_reward
        self.softmax = softmax
        self.sarsa = sarsa
        # initialize Q table
        # Still have to decide whether to shape it as [n_states*n_actions, n_actions] or [n_states, n_actions, n_actions]
        self.qtable = np.ones([n_states, n_actions, n_actions], dtype = float) * max_reward / (1 - discount)
        if qtable is not None:
            self.qtable = qtable

    def extract_state(self):
        state_dict = {
            'n_states' : self.nstates,
            'n_actions' : self.nactions,
            'discount' : self.discount,
            'max_reward' : self.max_reward,
            'softmax' : self.softmax,
            'sarsa' : self.softmax,
            'qtable' : self.qtable
        }
        return state_dict

    # action policy: implements epsilon greedy and softmax
    def select_action(self, state, epsilon):

        qval = self.qtable[state]
        prob = []
        if (self.softmax):
            # use Softmax distribution
            if epsilon == 0: epsilon = 1
            prob = sp.softmax(qval / epsilon)
        else:
            # assign equal value to all actions
            prob = np.ones(self.nactions) * epsilon / (self.nactions - 1)
            # the best action is taken with probability 1 - epsilon
            prob[np.argmax(qval)] = 1 - epsilon
        return np.random.choice(range(0, self.nactions), p = prob)
        
    # update function (Sarsa and Q-learning)
    def update(self, state, action, reward, next_state, alpha, epsilon):
        # find the next action (greedy for Q-learning, using the decision policy for Sarsa)
        next_action = self.select_action(next_state, 0)
        if (self.sarsa):
            next_action = self.select_action(next_state, epsilon)
        # calculate long-term reward with bootstrap method
        observed = reward + self.discount * self.qtable[next_state, next_action]
        # bootstrap update
        self.qtable[state, action] = self.qtable[state, action] * (1 - alpha) + observed * alpha