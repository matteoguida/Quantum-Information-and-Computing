'''
    Created on Oct 31st, 2020
    @author: Alberto Chimenti, Clara Eminente and Matteo Guida
    Purpose: (PYTHON3 IMPLEMENTATION)
        General purpose Q-learning approach for optimal quantum control
'''

#%%
from profiler_decorator import profile
import numpy as np
from environment import Environment
from Qmodel import quantum_model
import scipy.special as sp


class Agent:

    '''
    This class implements the Agent object. It saves various arguments among which the dimensions of the state-action space, 
    some internal learning parameters, the qtable and booleans for the choice of the behavioural policy.
    '''

    nsteps = 1
    nactions = 1
    discount = 1
    lmbda = 0.8
    qtable = np.matrix([1])
    softmax = True
    sarsa = False
    reward_bool = False
    
    # initialize
    def __init__(self, nsteps, nactions, qtable=None, **kwargs): 
        '''
        Initializes the Agent class

        INPUTS:
        nsteps: integer, the number of time steps per episode
        nactions: integer, the number of actions
        qtable: (optional) np.array(dtype=float) of size [nsteps*nactions, nactions], useful if one wants to import a pretrained q-table or force different initialization
        **kwargs: possible optional inputs
            discount: float, discount learning parameter gamma
            lambda: float, lambda parameter for eligibility trace update
            softmax: boolean, decides whether to use softmax behavioural policy or not
            sarsa: (old) boolean, decides whether to use off-policy algorithm version
        
        If qtable is not given as input, initalizes it together with the eligibility trace
        '''

        self.nsteps = nsteps
        self.nactions = nactions
        self.nstates = self.nsteps*self.nactions

        if 'discount' in kwargs:
            self.discount = kwargs.get('discount')
        if 'lambda' in kwargs:
            self.lmbda = kwargs.get('lambda')
        if 'softmax' in kwargs:
            self.softmax = kwargs.get('softmax')
        if 'sarsa' in kwargs:
            self.sarsa = kwargs.get('sarsa')

        self._init_qtable()
        self._init_trace()
        if qtable is not None:
            qtable = np.array(qtable)
            # Check whether the imported qtable has the correct shape
            if np.shape(qtable)==[self.nstates, self.nactions]:
                self.qtable = qtable
            else: 
                print("WARNING ----> Qtable size doesn't match given arguments \n [nstates*nactions, nactions]=", [self.nstates*self.nactions, self.nactions], "\n Given:", np.shape(qtable))

    def _init_qtable(self):
        '''
        Initializes Q table [The indexing will be index(t)*len(h)+index(h)]
        '''
        self.qtable = np.zeros([self.nstates, self.nactions], dtype = float)

    def _init_trace(self):
        '''
        Initializes Eligibility trace [The indexing will be index(t)*len(h)+index(h)]
        '''
        self.trace = np.zeros([self.nstates, self.nactions], dtype = float)

    def _init_evironment(self, model, starting_action, all_actions, history=True):
        '''
        Initializes the external environment class as an object
        ******dependent on external class*******

        INPUTS:
        model: custom_class object, model class object
        starting_action: integer, index corresponding to the starting action
        all_actions: list of integers, contains the possible action values (control field values)
        history: (optional) boolean, decides whether to store the path history or not
        '''
        self.env = Environment(model, starting_action, all_actions, history)

    def extract_state(self):
        '''
        Method used for agent feature extraction

        OUTPUT:
        state_dict: dictionary containing the features of the agent
        '''
        state_dict = {
            'nstates' : self.nstates,
            'nactions' : self.nactions,
            'discount' : self.discount,
            'softmax' : self.softmax,
            'sarsa' : self.sarsa,
            'qtable' : self.qtable
            }
        return state_dict

    # action policy: implements epsilon greedy and softmax
    def select_action(self, state, epsilon, greedy=False, replay=False):
        '''
        Selects action given the state and outputs the index of the chosen action

        INPUTS:
        state: integer, index of the current state
        epsilon: float, value of the epsilon parameter
        greedy: (optional) boolean, sets the action selection to greedy
        replay: (optional) boolean, forces the action selection to repropose the best protocol corresponding action

        OUTPUT:
        indA: integer, index of the selected action
        '''
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
                    # use Softmax policy
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
        '''
        Given the chosen action, moves the environment to the next state and updates the Q-table

        INPUTS:
        action: integer, next action index
        alpha: float, learning rate for update rule
        epsilon: float, epsilon parameter for off-policy selection of a_{t+1}
        '''
        # update trace
        self.trace[self.env.state.previous, self.env.state.action] = alpha

        observed = - self.qtable[self.env.state.previous, self.env.state.action] + self.env.reward

        if self.reward_bool:
            # for last time step iteration
            self.qtable += alpha * observed * self.trace
            return

        # calculate long-term reward with bootstrap method
        ###### DECISION POLICY #######
        # find the next action (greedy for Q-learning, epsilon-greedy for Sarsa)
        if (self.sarsa):
            next_action = self.select_action(self.env.state.current, epsilon)
        else:
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


    #@profile(sort_args=['name'], print_args=[80])
    def train_episode(self, starting_action, alpha, epsilon, replay=False):
        '''
        Trains the Agent for a given episode

        INPUTS:
        starting_action: integer, index of the starting action used to initialize the environment
        alpha: float, episode learning rate
        epsilon: float, episode epsilon parameter
        replay: (optional) boolean, decides whether to replay past optimal protocol or search for a new one
        '''
        # intialize environement
        self.env.reset(starting_action)
        self.env.model.reset()
        self._init_trace()
        self.protocol = []

        for step in range(self.nsteps):

            self.reward_bool = (step == self.nsteps - 1) #decides whether to compute reward or not

            # behavioural policy
            action = self.select_action(self.env.state.current, epsilon, replay=replay) #greedy=False by default

            # evolve quantum model
            self.env.model.evolve(self.env.all_actions[action])

            # move environement current ---> previous
            self.env.move(action, self.reward_bool)

            # append action to protocol
            self.protocol.append(self.env.all_actions[self.env.state.action])

            # update agent's Q-table
            self.update(action, alpha, epsilon)
            

    def train_agent(self, starting_action, episodes, alpha_vec, replay_freq, replay_episodes, verbose=False, epsilon_i=1, epsilon_f=0, conv_check=10):
        '''
        Simple wrapper for training procedure

        INPUTS:
        starting_action: integer, starting action index
        episodes: integer, number of episodes to run for training
        alpha_vec: tuple of floats of size [episodes], contains the learning rate used for each episode
        replay_freq: integer, number of episodes to run before each replay session
        replay_episodes: integer, number of replay episodes to run during replay session
        verbose: (optional) boolean, checks whether to print additional information or not
        epsilon_i: (optional) float, starting epsilon value for RB-epsiolon-D
        epsilon_f: (optional) float, final epsilon value for RB-epsiolon-D
        conv_check: (optional) integer, number of test protocols used at the end of the training to check Q-table convergence

        OUTPUTS:
        rewards: list of floats of size [episodes], contains the rewards obtained per episode
        mavg_rewards: list of floats of size [episodes+1], contains the incremental moving average over the obtained rewards
        epsilons: list of floats of size [episodes+1], contains the epsilons found during training with RB-epsilon-D

        '''
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
            if index%20==0:
                epsilon = self.update_greedyness(episodes, index, epsilon, mavg_rewards[-1])
            epsilons.append(epsilon)
            #############################

            #### BEST REWARD/PROTOCOL UPDATE ####
            if self.best_reward < self.env.reward:
                self.best_protocol = self.protocol
                self.best_reward = self.env.reward
                self.best_path = self.env.model.qstates_history
                if verbose:
                    print('\nNew best protocol {} with reward {}'.format(index, self.best_reward))

            # Replay episodes
            if index%replay_freq==0 and index!=0:
                if verbose:
                    print("\n...Running replay epidosdes...")
                for _ in range(replay_episodes):
                    self.train_episode(starting_action, alpha_vec[index], epsilon, replay=True)
                    rewards.append(self.env.reward)
                    mavg_rewards.append(((mavg_rewards[-1]*index) + self.env.reward)/(index+1))
                    #############################
                    epsilons.append(epsilon)
                    #############################

        # Last point test
        _, reward = self.generate_protocol(starting_action)
        rewards.append(reward)

        # Test convergence
        if conv_check is not None:
            print("----> Testing convergence...")
            test_sum = 0
            for _ in range(conv_check):
                _, reward = self.generate_protocol(starting_action)
                test_sum += reward
            error = np.abs(self.best_reward - test_sum/conv_check)
            if error > 1e-3:
                print("!WARNING: The Q-table does not converge. Deviating {} from best protocol fidelity".format(error))
            else:
                print("Learning seems to be fine!")

        return rewards, mavg_rewards, epsilons


    def update_greedyness(self, episodes, episode, epsilon, avg_reward, max_steps=10, T=8):
        '''
        Reward-based-epsilon-decay
        Updates the greediness parameter given the episode and the reward 

        INPUTS:
        episodes: integer, total number of episodes
        episode: integer, current episode index
        avg_reward: float, obtained reward
        max_steps: (optional) integer, decides how many steps to tolerate and how much increment to consider for reward threshold
        T: (optional) float, is the temperature factor in the exponential decay of the epsilon parameter

        OUTPUT:
        epsilon: float, epsilon value
        '''
        if (avg_reward >= self.avg_reward) or (self.counter >= max_steps):
            self.avg_reward += (avg_reward-self.avg_reward)*(max_steps/100) # adds 10% of the increment
            epsilon = self.epsilon_f + (self.epsilon_i - self.epsilon_f)*np.exp(-T*episode/episodes)
            self.counter = 0
        else:
            self.counter += 1
        return epsilon


    def generate_protocol(self, starting_action, **kwargs):
        '''
        Simple wrapper used to generate protocol given a trained agent

        INPUTS:
        starting_action: integer, starting action index
        **kwargs: possible option inputs
            qstart: np.array(dtype=complex) of size [2^{L}], quantum starting state for the model custom class object
        '''
        # intialize environement
        self.env.reset(starting_action)
        if 'qstart' in kwargs: 
            self.env.model.qstart = kwargs.get('qstart')
        self.env.model.reset()
        self.protocol = []

        for step in range(self.nsteps):

            self.reward_bool = (step == self.nsteps - 1) #decides whether to compute reward or not

            # behavioural policy
            action = self.select_action(self.env.state.current, 0, greedy=True) #greedy=False by default

            # evolve quantum model
            self.env.model.evolve(self.env.all_actions[action])

            # move environement current ---> previous
            self.env.move(action, self.reward_bool)

            # append action to protocol
            self.protocol.append(self.env.all_actions[self.env.state.action])

        return self.protocol, self.env.reward


def protocol_analysis(qstart, qtarget, t_max_vec, n_steps, all_actions, **kwargs):

    '''
    Wrapper function which runs analysis over the different maximum fidelity performances for different T_max

    INPUTS:
    qstart: np.array(dtype=complex), quantum starting state
    qtarget: np.array(dtype=complex), quantum target state
    t_max_vec: list of floats, contains the values of T_max to use for each iteration
    n_steps: integer, number of timesteps to consider for each episode
    all_actions: list of integers, contains the possible action values (control field values)
    **kwargs: possible optional inputs to change default values
        L: integer, size of the quantum system
        g: integer, static field value
        starting_action: integer, index of the starting action
        episodes: integer, number of episodes to run
        replay_freq: integer, number of episodes to run before each replay session
        replay_episodes: integer, number of replay episodes to run during replay session

    OUTPUT:
    fidelities: list of floats, containing the final fidelities obtained after training for each T_max
    '''

    # Default values
    L=1
    g=1
    starting_action = 0
    episodes = 20001
    replay_freq=50
    replay_episodes=40

    if 'L' in kwargs:
        L = kwargs.get('L')
        print("Overwritten default L with:", L)
    if 'g' in kwargs:
        g = kwargs.get('g')
        print("Overwritten default g with:", g)
    if 'starting_action' in kwargs:
        starting_action = kwargs.get('starting_action')
        print("Overwritten default starting_action with:", starting_action)
    if 'episodes' in kwargs:
        episodes = kwargs.get('episodes')
        print("Overwritten default episodes with:", episodes)
    if 'replay_freq' in kwargs:
        replay_freq = kwargs.get('replay_freq')
        print("Overwritten default replay_freq with:", replay_freq)
    if 'replay_episodes' in kwargs:
        replay_episodes = kwargs.get('replay_episodes')
        print("Overwritten default replay_episodes with:", replay_episodes)

    # alpha value
    a=0.9; eta=0.89
    alpha = np.linspace(a, eta, episodes)

    fidelities = []
    for t_max in t_max_vec:
        print("\n Running training for T={}".format(t_max))

        dt = t_max/n_steps
        model = quantum_model(qstart, qtarget, dt, L, g, all_actions)

        # initialize the agent
        learner = Agent(n_steps, len(all_actions))
        learner._init_evironment(model, starting_action, all_actions)
        # train
        _ = learner.train_agent(starting_action, episodes, alpha, replay_freq, replay_episodes, verbose=False)
        print("Found protocol with fidelity:", learner.best_reward)
        fidelities.append([t_max, learner.best_reward])
    
    return fidelities



############################################################################


if __name__ == "__main__":

    from Qmodel import quantum_model, ground_state
    from pathlib import Path

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
    h_list=[-4,0,4]
    print("\nRunning analysis for L="+str(L))
    print("\nFor T_max:\n", times)

    fidelities = protocol_analysis(qstart, qtarget, times, n_steps, h_list, L=L)
    fname = "fidelity_RL_L_"+str(L)+".txt"
    fname = out_dir / fname
    np.savetxt(fname, fidelities, delimiter = ',')

