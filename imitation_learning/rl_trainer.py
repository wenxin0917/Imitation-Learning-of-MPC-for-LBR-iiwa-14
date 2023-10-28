import time
from collections import OrderedDict
import pickle
import numpy as np
import gym
import os
import torch
import random

from utils import *
from imitation_learning.logger import Logger
from environment.gym_env import iiwa14Env,iiwa14EnvOptions
from imitation_learning.bc_agent import BCAgent
from imitation_learning.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from expert_data.generate_expert import *


class RL_Trainer(object):
    def __init__(self,params) -> None:
        
        # get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])
        
        # set random seeds , it helps to reproduce the results
        seed = self.params['seed']
        torch.manual_seed(seed)
        np.random.seed(seed) # because the seed is stored, so the result will always be the same as long as the seed is the same
        
        # make the gym environments
        self.env = iiwa14Env(iiwa14EnvOptions(dt= 0.05,x_start=None,x_end=None,sim_time=None,sim_noise_R=None,contr_input_state=None))
        
        """
        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.shape[0]
        """
        
        # create the agent
        self.agent = BCAgent(self.env, self.params)
        # create the data buffer
        self.buffer = ReplayBuffer()
    
    def run_training_loop_bc(self, n_iter,initial_expert_state=None,initial_expert_action=None):
       
        """
        :param n_iter:  number of (dagger) iterations
        :param initial_expert_state/action : 
        """
        
        optimizer = self.agent.actor.optimizer
        
        training_loss = []
        validation_loss = []
        # init vars at beginning of training
        self.start_time = time.time()
        scheluer = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)
            self.log_metrics = True
            # train agent (using sampled data from replay buffer)
            total_loss= self.train_agent_bc(initial_expert_state,initial_expert_action) 
            
            # print("[EPOCH]: %i, [MSE LOSS]: %.6f" % (itr, total_loss))
            # training_loss.append(total_loss)
            print("[EPOCH]: %i, [MSE LOSS]: %.6f" % (itr, total_loss / self.params['num_agent_train_steps_per_iter']))
            training_loss.append(total_loss / self.params['num_agent_train_steps_per_iter'])
            validation_loss.append(self.validation_process())
            scheluer.step()
            
        print("\nSaving agent's actor...")
        self.agent.actor.save(self.params['logdir'] + '/try2_policy_itr_'+str(itr)+'.pth')

        # save the list to a file
        with open('training_logger/training_loss_try_2.pkl','wb') as file:
            pickle.dump(training_loss,file)
        with open('training_logger/validation_loss_try_2.pkl','wb') as file:
            pickle.dump(validation_loss,file)
            
            
            
    def train_agent_bc(self,state,action):
        print('\nTraining agent using data from trainingset...')
        total_loss = 0
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            start_index = train_step * self.params['batch_size']
            end_index = (train_step + 1) * self.params['batch_size']

            # sample some data from the expert data
            ob_batch = state[start_index:end_index,:]
            ac_batch = action[start_index:end_index,:]
            # use the sampled data for training
            total_loss += self.agent.train(ob_batch, ac_batch)
            
        # total  loss represent the whole loss in one epoch    
        return total_loss
    
    def run_training_loop_dagger(self, n_iter):
       
        """
        :param n_iter:  number of (dagger) iterations
        :param initial_expert_state/action : 
        """
        
        optimizer = self.agent.actor.optimizer
        
        training_loss = []
        validation_loss = []
        # init vars at beginning of training
        self.start_time = time.time()
        scheluer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

        for itr in range(n_iter):
            
            print("\n\n********** Iteration %i ************"%itr)
            # train agent (using sampled data from replay buffer)
            total_loss= self.train_agent_dagger(itr)
            print("[EPOCH]: %i, [MSE LOSS]: %.6f" % (itr, total_loss / self.params['num_agent_train_steps_per_iter']))
            training_loss.append(total_loss / self.params['num_agent_train_steps_per_iter'])
            validation_loss.append(self.validation_process())
            scheluer.step()
            
            """
            # save the model
            if (itr+1) % 5 == 0:
                # save policy
                print("\nSaving agent's actor...")
                self.agent.actor.save(self.params['logdir'] + '/dagger1_policy_itr_'+str(itr)+'.pth')
            """
            
        # save policy
        print("\nSaving agent's actor...")
        self.agent.actor.save(self.params['logdir'] + '/try2_policy_itr_'+str(itr)+'.pth')
        # print("current learning rate is : ",optimizer.param_groups[0]["lr"])
            
        # save the list to a file
        with open('training_logger/training_loss_try_2.pkl','wb') as file:
            pickle.dump(training_loss,file)
        with open('training_logger/validation_loss_try_2.pkl','wb') as file:
            pickle.dump(validation_loss,file)
            
    
    def train_agent_dagger(self,iteration):
        print('\nTraining agent using data from trainingset...')
        total_loss = 0
        state,action = self.buffer.get_data()
        print("current training data size: ",state.shape[0])
        
        if state.shape[0] % self.params['batch_size'] == 0:
            self.params['num_agent_train_steps_per_iter'] = state.shape[0] // self.params['batch_size']
        else:
            self.params['num_agent_train_steps_per_iter'] = (state.shape[0] // self.params['batch_size']) + 1
        
        if (iteration+1) % 50 == 0:
            self.mix_nn_output_with_expert(self.agent,iteration)
            state,action = self.buffer.get_data()
            print("current training data size: ",state.shape[0])
        # add new training data
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            start_index = train_step * self.params['batch_size']
            end_index = (train_step + 1) * self.params['batch_size']

            # sample some data from the expert data
            ob_batch = state[start_index:end_index,:]
            ac_batch = action[start_index:end_index,:]
            # use the sampled data for training
            total_loss += self.agent.train(ob_batch, ac_batch)
        # total  loss represent the whole loss in one epoch    
        return total_loss
    
     
    def validation_process(self):
        validation_states = np.load('expert_data/validation_states.npy')
        validation_actions = np.load('expert_data/validation_actions.npy')
        total_loss = 0
        if validation_actions.shape[0] % self.params['batch_size'] == 0:
            num_agent_validation_steps = validation_actions.shape[0] // self.params['batch_size']
        else:
            num_agent_validation_steps = (validation_states.shape[0] // self.params['batch_size']) + 1
        
        for val_step in range(num_agent_validation_steps):
            start_index = val_step * self.params['batch_size']
            end_index = (val_step + 1) * self.params['batch_size']

            ob_batch = validation_states[start_index:end_index, :]
            ac_batch = validation_actions[start_index:end_index, :]

            # Use the sampled validation data for computing loss
            total_loss += self.agent.compute_loss(ob_batch, ac_batch)

        # return total_loss
        return total_loss / num_agent_validation_steps
     

    def mix_nn_output_with_expert(self,agent,itr):
        
        print("\nMixing current policy with expert policy to add more data")

        # 20 is the number of iterations we want to mix the expert policy with our policy
        # 20 = 500(total_iteration) / 25
        if itr+50 > 2000 or itr < 300:
            return
        # else:
            # if itr == 0:   
                # t =  1
            # else:
        t = (itr+51)/50
        mixture_coefficient = t / 40
        
        q_range = np.deg2rad([170,120,170,120,170,120,175])
        alpha = 0.3   
        np.random.seed(int(t))
        for i in range(100):
            x_0 = sample_rand_initial_position(alpha,q_range)
            print("initial state",x_0)
            explored_states, explored_actions,explored_output = generate_expert_data_with_NN(mixture_coefficient,agent,x_0)

            if np.all(explored_actions == np.zeros((7,))):
                continue
            else:   
                self.buffer.add_data(explored_states,explored_actions)
                new_states,new_actions = self.buffer.get_data()
                np.save("expert_data/dagger_states_2.npy",new_states)
                np.save("expert_data/dagger_actions_2.npy",new_actions)

        return 