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
        self.env = iiwa14Env(iiwa14EnvOptions(dt= None,x_start=None,x_end=None,sim_time=None,sim_noise_R=None,contr_input_state=None))
        
        """
        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.shape[0]
        """
        
        # create the agent
        self.agent = BCAgent(self.env, self.params)
    
    def run_training_loop_bc(self, n_iter,initial_expert_state=None,initial_expert_action=None):
       
        """
        :param n_iter:  number of (dagger) iterations
        :param initial_expert_state/action : 
        """

        # init vars at beginning of training
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)
            self.log_metrics = True
            """
            # decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False
            """
            # train agent (using sampled data from replay buffer)
            total_loss= self.train_agent_bc(initial_expert_state,initial_expert_action) 
            print("[EPOCH]: %i, [MSE LOSS]: %.6f" % (itr+1, total_loss / self.params['num_agent_train_steps_per_iter']))

            # save the best model
            if self.log_metrics:
                # save policy
                print("\nSaving agent's actor...")
                self.agent.actor.save(self.params['logdir'] + '/policy_itr_'+str(itr)+'.pth')
    
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
     
     
     
      
    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expert_state=None,initial_expert_action=None,relabel_with_expert=None,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expert_state/action : 
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr+1)
            
            # decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(itr,
                                initial_expert_action,initial_expert_state, collect_policy,
                                self.params['batch_size'])
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            self.train_agent() 

            # log/save
            if self.log_metrics:

                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths)

                # save policy
                print('\nSaving agent\'s actor...')
                self.agent.actor.save(self.params['logdir'] + '/policy_itr_'+str(itr))

            
            
    def collect_training_trajectories(self, itr, load_initial_expert_action,load_initial_state, collect_policy, batch_size):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file or npy file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # if itr == 0:
            # ??? how to load expert data according the file path and file type
            # with open(load_initial_expertdata, "rb") as f:
                # loaded_paths = pickle.load(f)
            # return loaded_paths, 0, None

        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'])

        return paths, envsteps_this_batch
    
    
    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            #For logging within an episode
            #if train_step % 2000 == 0:
            #    self.perform_logging(train_step, None, self.agent.actor, None)

            # sample some data from the data buffer
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            
            # use the sampled data for training
            self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

    def do_relabel_with_expert(self, expert_policy, current_nn_output,observation):
        print("\nRelabelling collected observations with labels from an expert policy...")

        # relabel collected obsevations (from our policy) with labels from an expert policy
        mix_output = self.params['belta']*(expert_policy.get_action(observation)) + (1-self.params['belta'])*current_nn_output
        return mix_output