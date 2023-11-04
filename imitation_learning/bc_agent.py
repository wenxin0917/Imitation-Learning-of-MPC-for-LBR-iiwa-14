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
        
        """
        # initialize for BC 
        self.actor = MLPPolicy(
            self.params['ob_dim'],
            self.params['ac_dim'],
            self.params['n_layers'],
            self.params['size'],
            self.params['device'],
            self.params['learning_rate']
        )
        """
        
        """
        # load the trained bc model for Dagger training
        self.actor = MLPPolicy(14,7,6,256,device='cpu',lr=0.001,training= True)
        checkpoint = torch.load('training_logger/bc4_policy_itr_999.pth')
        self.actor.load_state_dict(checkpoint)
        """
        
        
        # replay buffer to store data
        self.replay_buffer = ReplayBuffer(
            self.params['max_replay_buffer_size']
        )
        
    def train(self,ob_no,ac_na):
        # update/fit actor/policy
        loss =self.actor.update(ob_no,ac_na)
        return loss
    
    def compute_loss(self,ob_no,ac_na):
        
        # update/fit actor/policy
        loss =self.actor.compute_loss(ob_no,ac_na)
        return loss
    
    def get_action(self,obs):
        # query actor/policy for action given observation(s)
        return self.actor.get_action(obs)