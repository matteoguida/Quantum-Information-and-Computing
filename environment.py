'''
    Created on Oct 30th, 2020

    @author: Alberto Chimenti

    Purpose: (PYTHON3 IMPLEMENTATION)
'''

import numpy as np

class state_object(object):
    """ Generic reinforcement learning object """

    def __init__(self):
        self.initial = None
        self.previous = None
        self.action = None # Action(index) which moved previous-->current
        self.current = None
        self.visited = []

class Environment(object):

    def __init__(self, model, starting_action, all_actions=[-4, 0, +4], history=True):

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
        # Maps action index into state indexing for Q-table
        if t == 'next':
            return (self.time_step + 1)*len(self.all_actions) + action_idx
        elif t == 'previous':
            return (self.time_step - 1)*len(self.all_actions) + action_idx
        else:
            return self.time_step*len(self.all_actions) + action_idx
    
    def state_action_map(self, state, time_step):
        # Maps state index into action index
        return state - time_step*len(self.all_actions)


    def move(self, action, final_bool):

        # move state and save previous values given a new action
        self.state.previous = self.state.current

        self.time_step += 1 #increment time_step

        self.state.action = action
        self.state.current = self.action_state_map(action)

        if self.history:
            self.state.visited.append(self.state.current)
        
        # Compute model reward (possibly for last time step in episode)
        if final_bool:
            self.reward = self.model.compute_fidelity()
        