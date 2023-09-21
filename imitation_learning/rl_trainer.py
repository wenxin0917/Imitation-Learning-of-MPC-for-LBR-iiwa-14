import time
from collections import OrderedDict
import pickle
import numpy as np
import gym
import os
import torch

from utils import *
from imitation_learning.logger import Logger
from environment.gym_env import iiwa14Env,iiwa14EnvOptions
from imitation_learning.bc_agent import BCAgent
from imitation_learning.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt



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
        scheluer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)
            self.log_metrics = True
            # train agent (using sampled data from replay buffer)
            total_loss= self.train_agent_bc(initial_expert_state,initial_expert_action) 
            print("[EPOCH]: %i, [MSE LOSS]: %.6f" % (itr, total_loss / self.params['num_agent_train_steps_per_iter']))
            training_loss.append(total_loss / self.params['num_agent_train_steps_per_iter'])
            validation_loss.append(self.validation_process())
            scheluer.step()
            
        print("\nSaving agent's actor...")
        self.agent.actor.save(self.params['logdir'] + '/bc26_policy_itr_'+str(itr)+'.pth')

        # save the list to a file
        with open('training_logger/training_loss_bc_26.pkl','wb') as file:
            pickle.dump(training_loss,file)
        with open('training_logger/validation_loss_bc_26.pkl','wb') as file:
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
        scheluer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

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
        self.agent.actor.save(self.params['logdir'] + '/dagger3_policy_itr_'+str(itr)+'.pth')
        # print("current learning rate is : ",optimizer.param_groups[0]["lr"])
            
        # save the list to a file
        with open('training_logger/training_loss_dagger_3.pkl','wb') as file:
            pickle.dump(training_loss,file)
        with open('training_logger/validation_loss_dagger_3.pkl','wb') as file:
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
        
        if iteration ==0 or (iteration+1) % 50 == 0:
            self.mix_nn_output_with_expert(self.agent,action,state,iteration)
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

        return total_loss / num_agent_validation_steps
     
    
    

    def mix_nn_output_with_expert(self, current_policy, expert_output,observation,itr):
        
        print("\nMixing current policy with expert policy to add more data")

        # 20 is the number of iterations we want to mix the expert policy with our policy
        # 20 = 500(total_iteration) / 25
        
        if itr+50 >=999:
            return
        else:
            t = (itr+50) / 50 
            mixture_coefficient = t / 20
            
        # mixture_coefficient = 1/30
        # relabel collected obsevations (from our policy) with labels from an expert policy
           
        # Randomly choose 200 indices without replacement
        chosen_indices = np.random.choice(observation.shape[0], size=200, replace=False)
        new_observation = observation[chosen_indices, :]
        mix_output = mixture_coefficient*(current_policy.get_action(new_observation)) + (1-mixture_coefficient)*expert_output[chosen_indices, :]
        added_new_state = np.zeros((mix_output.shape[0],14))
        for i in range(mix_output.shape[0]):
            added_new_state[i,:] = self.env.simulator.integrator_step(observation[i,:],mix_output[i,:],).reshape(-1,14)
        
        # print(added_new_state)
        # print(added_new_state.shape)
        self.buffer.add_data(added_new_state,None)
        new_states,new_actions = self.buffer.get_data()
        np.save("expert_data/dagger_train_states_3.npy",new_states)
        np.save("expert_data/dagger_train_actions_3.npy",new_actions)

        return 
        # return added_new_state