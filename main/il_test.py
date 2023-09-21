import torch
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from imitation_learning.MLP_policy import MLPPolicy
from environment.gym_env import iiwa14EnvOptions,iiwa14Env
from model.iiwa14_model import Symbolic_model
import numpy as np

import matplotlib.pylab as plt

def plot_actions(predict_actions,expert_actions):
    t = np.arange(0, 41) * 0.05
    # Plot and label the training and validation loss values
    u_lbls = [fr'$u_{k}$ [N*m]' for k in range(1, 8)]
    _, axs_u = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    for k, ax in enumerate(axs_u.reshape(-1)):
        ax.step(t[:-1], predict_actions[:, k], label='predicted_actions',color='blue')
        ax.step(t[:-1], expert_actions[:, k],label='expert_actions',color='orange')
        ax.set_ylabel(u_lbls[k])
        ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    
def plot_measurements(t: np.ndarray, y: np.ndarray, pee_ref: np.ndarray = None, axs=None):
    # y output (1,17)
    # Parse measurements
    q = y[:, :7]
    dq = y[:, 7:14]
    pee = y[:, 14:]

    q_lbls = [fr'$q_{k}$ [rad]' for k in range(1, 8)]
    do_plot = True
    if axs is None:
        _, axs_q = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    else:
        axs_q = axs
        do_plot = False
    for k, ax in enumerate(axs_q.reshape(-1)):
        ax.plot(t, q[:, k])
        ax.set_ylabel(q_lbls[k])
        ax.grid(alpha=0.5)
    axs_q[2].set_xlabel('t [s]')
    plt.tight_layout()

    dq_lbls = [fr'$\dot q_{k}$ [rad/s]' for k in range(1, 8)]
    _, axs_dq = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    for k, ax in enumerate(axs_dq.reshape(-1)):
        ax.plot(t, dq[:, k])
        ax.set_ylabel(dq_lbls[k])
        ax.grid(alpha=0.5)
    plt.tight_layout()

    pee_lbls = [r'$p_{ee,x}$ [m]', r'$p_{ee,y}$ [m]', r'$p_{ee,z}$ [m]']
    _, axs_pee = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
    for k, ax in enumerate(axs_pee.reshape(-1)):
        ax.plot(t, pee[:, k])

        ax.axhline(pee_ref[k], ls='--', color='red')

        ax.set_ylabel(pee_lbls[k])
        ax.grid(alpha=0.5)
    plt.tight_layout()
    if do_plot:
        plt.show()            
            

test_states = np.load('expert_data/test_states.npy')
test_data = test_states.reshape(-1,40,14)
test_actions = np.load('expert_data/test_actions.npy')
test_actions = test_actions.reshape(-1,40,7)
reward = []
x_ref = np.array([-0.44,0.14,2.02,-1.61,0.57,-0.16,-1.37,0,0,0,0,0,0,0]).reshape(14,1)

# load the model
loaded_model = MLPPolicy(14,7,6,256,device='cpu',lr=0.001,training= False)
checkpoint = torch.load('training_logger/dagger2_policy_itr_1999.pth')
loaded_model.load_state_dict(checkpoint)

# load the iiwa14 model 
iiwa14_model = Symbolic_model()
pee_ref = iiwa14_model.forward_kinemetics(x_ref[:7])
time = np.arange(0,41)*0.05

with torch.no_grad():
        for i in range(test_data.shape[0]):
            input_array = test_data[i,0,:]
            env_options = iiwa14EnvOptions(dt = 0.05,x_start = input_array.reshape(14,1),x_end = x_ref,sim_time = 2,sim_noise_R = None,contr_input_state = 'real')
            env = iiwa14Env(env_options)
            env.reset()
            inputs = torch.Tensor(input_array)
            done = False
            sum_reward = 0
            predict_actions = np.zeros((1,7))
            initial_pee = iiwa14_model.forward_kinemetics(input_array[:7])
            state = input_array
            outputy = np.hstack((state.reshape(1,14),initial_pee.reshape(1,3)))
            while(done == False): 
                predictions = loaded_model(inputs)
                predictions = predictions.numpy()
                
                predict_actions= np.vstack((predict_actions,predictions.reshape(1,7)))
                
                (next_input,step_reward,done,_) = env.step(predictions.reshape(1,7))
                next_pee = iiwa14_model.forward_kinemetics(next_input[:7])
                state = np.vstack((state,next_input.reshape(1,14)))
                current_y = np.hstack((next_input.reshape(1,14),next_pee.reshape(1,3)))
                print("current_y shape:",current_y.shape)
                outputy = np.vstack((outputy,current_y))
                sum_reward += step_reward
                inputs = torch.Tensor(next_input)
            plot_actions(predict_actions[1:,:],test_actions[i,:,:].reshape(40,7))   
            plot_measurements(time,outputy,pee_ref)
                
            reward.append(sum_reward)

print(reward)
