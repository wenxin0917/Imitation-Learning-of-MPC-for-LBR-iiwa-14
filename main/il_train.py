import numpy as np
import torch
import os
import sys
import pickle
import matplotlib.pylab as plt
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from sklearn.model_selection import train_test_split
from imitation_learning.rl_trainer import RL_Trainer
from imitation_learning.bc_agent import BCAgent

class BC_Trainer(object):
    def __init__(self,params) -> None:
        
        #######################
        ## AGENT PARAMS
        #######################
        self.params = params
        
        ################
        ## RL TRAINER
        ################
        self.rl_trainer = RL_Trainer(self.params)
        
        #######################
        ## LOAD EXPERT ACTION AND POLICY
        #######################
        
        
        # print('Loading expert policy from...', self.params['expert_policy_file'])
        # self.expert_policy = MPC()
        # print('Done restoring expert action...')
    
    
    def run_training_loop_bc(self,expert_state,expert_action):
        """
            Implement the main training loop for BC.
            Loop should terminate when any of the following conditions are met:
                - max time steps exceeded
                - max iterations exceeded
                - performance exceeds threshold
        """
        
        self.rl_trainer.run_training_loop_bc(
            n_iter = self.params['n_iter'],
            initial_expert_state= expert_state,
            initial_expert_action= expert_action
        )
        
class DAgger_Trainer(object):
    def __init__(self,params) -> None:
        
        #######################
        ## AGENT PARAMS
        #######################
        self.params = params
        
        ################
        ## RL TRAINER
        ################
        self.rl_trainer = RL_Trainer(self.params)
        initial_state = np.load('expert_data/train_states.npy')
        initial_action = np.load('expert_data/train_actions.npy')
        self.rl_trainer.buffer.add_data(initial_state,initial_action)
    
    
    def run_training_loop_dagger(self):
        """
            Implement the main training loop for BC.
            Loop should terminate when any of the following conditions are met:
                - max time steps exceeded
                - max iterations exceeded
                - performance exceeds threshold
        """
        
        self.rl_trainer.run_training_loop_dagger(
            n_iter = self.params['n_iter'],
        )      




def bc_main(depth,width,learning_rate,seed,states,actions):
    # directory_name = 'depth_'+str(depth)+'_width_'+str(width)+'_lr_'+str(learning_rate)+'_seed_'+str(seed)
    # directory_path = os.path.join('training_logger', directory_name)
    # os.mkdir(directory_path)
    params = {
        'n_layers': depth,
        'size': width,
        'learning_rate': learning_rate,
        'max_replay_buffer_size': 1000000,
        'n_iter': 500,
        'batch_size': 32,
        'agent_class': BCAgent,
        'do_gagger': False,
        'ep_length': 40,
        'logdir': 'training_logger',
        'seed': seed,
        'scalar_log_freq': 1,
        'num_agent_train_steps_per_iter':125, # data number / batch_size
        'ac_dim': 7,
        'ob_dim': 14,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    bc_trainer = BC_Trainer(params)
    bc_trainer.run_training_loop_bc(states,actions)


def dagger_main():
    # directory_name = 'depth_'+str(depth)+'_width_'+str(width)+'_lr_'+str(learning_rate)+'_seed_'+str(seed)
    # directory_path = os.path.join('training_logger', directory_name)
    # os.mkdir(directory_path)
    params = {
        'n_layers': 6,
        'size': 256,
        'learning_rate': 0.001,
        'max_replay_buffer_size': 1000000,
        'n_iter': 2000,
        'batch_size': 32,
        'agent_class': BCAgent,
        'do_gagger': False,
        'ep_length': 40,
        'logdir': 'training_logger',
        'seed': 20,
        'scalar_log_freq': 1,
        'num_agent_train_steps_per_iter':100, # data number / batch_size
        'belta': 0.5,
        'ac_dim': 7,
        'ob_dim': 14,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    dagger_trainer = DAgger_Trainer(params)
    dagger_trainer.run_training_loop_dagger()
       
def plot_training_validation_loss():
    
    # Load the list from the file
    with open('training_logger/training_loss_dagger_1.pkl', 'rb') as file:
        train_values = pickle.load(file)
    # with open('training_logger/validation_loss_try_2.pkl', 'rb') as file:
        # val_values = pickle.load(file)
        
    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, len(train_values) + 1)
 
    # Plot and label the training and validation loss values
    plt.plot(epochs, train_values, label='Training Loss')
    # plt.plot(epochs, val_values, label='Validation Loss')
 
    # Add in a title and axes labels
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
 
    # Set the tick locations
    # plt.xticks(arange(0, 21, 2))
 
    # Display the plot
    plt.legend(loc='best')
    plt.show()
        
if __name__ == "__main__":
    
    initial_state = np.load('expert_data/train_states.npy')
    initial_action = np.load('expert_data/train_actions.npy')
    bc_main(6,256,0.001,10,initial_state,initial_action)
    # dagger_main()
    plot_training_validation_loss()
    