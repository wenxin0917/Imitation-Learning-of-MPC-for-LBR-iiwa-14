import numpy as np
import time

from imitation_learning.MLP_policy import *
from imitation_learning.utils import *
from imitation_learning.replay_buffer import ReplayBuffer

class BCAgent:
    def __init__(self, env, params):
        # init vars
        self.env = env
        self.params = params
        # actor/policy
        self.actor = MLPPolicy(
            self.params['ob_dim'],
            self.params['ac_dim'],
            self.params['n_layers'],
            self.params['size'],
            self.params['device'],
            self.params['learning_rate']
        )
        
        # replay buffer to store data
        self.replay_buffer = ReplayBuffer(
            self.params['max_replay_buffer_size']
        )
        
    def train(self,ob_no,ac_na):
        # update/fit actor/policy
        loss =self.actor.update(ob_no,ac_na)
        return loss
    
    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)
    # ??? where can I get the paths???
    
    
    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)