'''
    Created on Oct 30th, 2020

    @author: Alberto Chimenti

    Purpose: (PYTHON3 IMPLEMENTATION)
'''

import numpy as np

class state_object(object):

    def __init__(self):
        self.initial = None
        self.previous = None
        self.current = None
        self.visited = []

class Environment(object):

    def __init__(self, starting_action=0.0, all_actions=[-4, 0, +4], history=True):

        self.history = history

        self.all_actions = all_actions

        self.action_map_dict = {all_actions[idx] : idx for idx in range(len(self.all_actions))}
        
        self.reset(starting_action)


    def reset(self, starting_action=0):

        # resets environment
        self.state = state_object() #Saved as indexed quantity for Q-table indexing
        self.action = state_object() #Saved as actual mag_field value for easier output usability
        self.time_step = 0 #Important for action--->state indexing (see. map_state method)

        self.state.initial = self.state_map(starting_action)
        self.action.initial = starting_action

        self.state.current = self.state.initial
        self.action.current = self.action.initial

        self.reward = 0.0

        if self.history:
            self.state.visited.append(self.state.initial)
            self.action.visited.append(self.action.initial)


    def state_map(self, action):
        # Maps action into state indexing for Q-table
        return self.time_step*len(self.all_actions) + self.action_map_dict[action]


    def move(self, action, time_ev_func):

        # move state and save previous values given a new action
        self.state.previous = self.state.current
        self.action.previous = self.action.current

        self.action.current = action
        self.state.current = self.state_map(action)

        self.time_step += 1

        if self.history:
            self.state.visited.append(self.state.current)
            self.action.visited.append(self.action.current)
        