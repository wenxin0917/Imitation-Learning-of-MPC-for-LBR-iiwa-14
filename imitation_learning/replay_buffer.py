import time
import numpy as np
import gym
import os

from utils import *

class ReplayBuffer(object):
    
    def __init__(self, max_size=1e6):
        
        self.max_size = max_size
        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.acs = None
        
    def __len__(self):
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0
    
    
    def add_data(self, state, expert_action):
        """
        Add (state, expert action) pair to the data buffer.
        """
        
        if self.acs is None:
            # initialize buffer
            self.obs = state
            self.acs = expert_action
            
        else:
            self.obs = np.concatenate([self.obs, state], axis=0)
            self.acs = np.concatenate([self.acs, expert_action], axis=0)

    def get_data(self):
        """
        Get the stored data from the buffer.
        """
        return self.obs, self.acs
        
if __name__ == '__main__':
    
    new_buffer = ReplayBuffer()
    # first time add data
    new_state = np.load('expert_data/train_states.npy')
    expert_action = np.load('expert_data/train_actions.npy')
    new_buffer.add_data(new_state[:10,:], expert_action[:10,:])
    states, expert_actions = new_buffer.get_data()
    print(states.shape)
    # second time add data
    new_buffer.add_data(new_state[11:21,:], expert_action[11:21,:])
    states, expert_actions = new_buffer.get_data()
    print(states.shape)