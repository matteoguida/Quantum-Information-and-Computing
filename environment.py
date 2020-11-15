'''
    Created on Oct 30th, 2020
    @authors: Alberto Chimenti, Clara Eminente and Matteo Guida.
    Purpose: (PYTHON3 IMPLEMENTATION)
        Methods and class to create an RL environment and interface with the underlying model.
'''

import numpy as np

class state_object(object):
    """ 

    Generic reinforcement learning object.
    This Class contains all the elements necessary to keep track of the moves withinf the environment of an RL 
    agent in a Q-Learning algorithm.

    """

    def __init__(self):
        self.initial = None
        self.previous = None
        self.action = None # Action(index) which moved previous-->current
        self.current = None
        self.visited = [] #List of visited states

class Environment(object):

    '''
    This Class defines the environment of an RL agent along with all the operations on it such as accessing states and actions given some variables
    (e.g. action_state_map, state_action_map) and moving in the environment (e.g. move).
    It's important to undeline that the index of a state in this framework is obtained combining two informations: the time and the values of the action
    at that timestep i.e. s = (t,a).


    INITIALIZATION VARIABLES:
    model: the model underlying the learning. It is only osed to compute the reward.
    starting_action: integer, indexed version of the action to start with
    all_actions: all possible actions
    history: boolean, if True keeps track of the visited states

    action_map_dict: dictionary, contains couples [action : action_index]
                    e.g. if all_actions is [-4,4] it is {-4 : 0 ; 4 : 1}
    
    
    '''


    def __init__(self, model, starting_action, all_actions=[-4, +4], history=True):

        self.history = history

        self.all_actions = all_actions

        self.action_map_dict = {all_actions[idx] : idx for idx in range(len(self.all_actions))}

        self.model = model
        
        self.reset(starting_action)


    def reset(self, starting_action=0):

        # resets environment

        self.state = state_object() #Saved as indexed quantity for Q-table indexing

        self.time_step = 0 #Important for action--->state indexing (see. map_state method)

        self.state.initial = self.action_state_map(starting_action)

        self.state.current = self.state.initial

        self.reward = 0.0

        if self.history:
            self.state.visited.append(self.state.current)


    def action_state_map(self, action_idx, t=None):
        '''
        This function maps action index into a state indexing for accessing the right Q-table entry. 
        Returns an index to access the Q-Table with.
        
        INPUTS:
        action_idx: integer, indexed version of the action
        t: str, if "None" the index corresponding to the pair [time, action] is computed. If "previous" ("next" respectively)
        the index corresponding to the pair [t-1, action] ([t+1, action], respectively) is computed.

        OUTPUTS:
        index: integer

        '''
        if t == 'next':
            return (self.time_step + 1)*len(self.all_actions) + action_idx
        elif t == 'previous':
            return (self.time_step - 1)*len(self.all_actions) + action_idx
        else:
            return self.time_step*len(self.all_actions) + action_idx
    
    def state_action_map(self, state, time_step):
        '''
        This function maps state index into an action index (i.e. starting from a state retrieves the corresponding action).
        This is basically the inverse of action_state_map.

        INPUTS:
        state: integer, index of the state
        time_step: integer, index of the time step

        OUTPUTS:
        action_idx: integer, index corresponding to the action
        
        '''

        return state - time_step*len(self.all_actions)


    def move(self, action, final_bool):

        '''
        Given an action and the current state, the function moves the environment to the new state and computes 
        the reward for the episode (only if the end of episode is reached)

        INPUTS:
        action: iteger, index of the action taken
        final_bool: boolean, if True the reward is computed and stored   

        '''

        # move state and save previous values given a new action

        #current state becomes previous
        self.state.previous = self.state.current

        self.time_step += 1 #increment time_step

        self.state.action = action
        #current state is computed according to timestep and action taken
        self.state.current = self.action_state_map(action)

        #history is updated
        if self.history:
            self.state.visited.append(self.state.current)
        
        # Compute model reward (if the end of the episode is reached)
        if final_bool:
            self.reward = self.model.compute_fidelity()
        