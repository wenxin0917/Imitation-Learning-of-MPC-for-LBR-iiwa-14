import numpy as np
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import random   
from model.iiwa14_model import Symbolic_model

from utils.plotting import plot_circle_3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.plotting import plot_circle_xyz,plot_torque,plot_q_dq
from utils.plotting import *
# state = np.load('expert_data/train_states.npy')

"""
def has_duplicate_rows(arr):
    arr_as_set = {tuple(row) for row in arr}
    return len(arr_as_set) < len(arr)

state = np.load('expert_data/6000training_state_0.3.npy')
new_state = np.load('expert_data/state_0.3.npy')
state = np.vstack((state,new_state.reshape(-1,14)))

for i in range(1,11):
    state = np.load('expert_data/{}state_0.3.npy'.format(i))
    action = np.load('expert_data/{}action_0.3.npy'.format(i))
    output = np.load('expert_data/{}output_0.3.npy'.format(i))
    if i == 1:
        np.delete(state,40,axis=1)
        new_state = state.reshape(-1,14)
        new_action = action.reshape(-1,7)
        new_output = output.reshape(-1,17)
    else:
        np.delete(state,40,axis=1)
        new_state = np.vstack((new_state,state.reshape(-1,14)))
        new_action = np.vstack((new_action,action.reshape(-1,7)))
        new_output = np.vstack((new_output,output.reshape(-1,17)))

np.save('expert_data/6000training_state_0.3.npy',new_state)
np.save('expert_data/6000training_action_0.3.npy',new_action)
np.save('expert_data/6000training_output_0.3.npy',new_output)

if has_duplicate_rows(state):
    print("yes")
else:
    print("no")
"""
 
"""
output = np.load('expert_data/6000expert_output_0.3.npy')
out_action = np.load('expert_data/6000expert_action_0.3.npy')
reshape_output = output.reshape(-1,41,17)
print(reshape_output.shape)
random_index = []
for i in range(0,200):
    np.random.seed(i)
    index = np.random.randint(0,6000)
    random_index.append(index)
    
t = np.arange(0, 40 +1) * 0.05
x_ref = np.array([-0.44,0.14,2.02,-1.61,0.57,-0.16,-1.37,0,0,0,0,0,0,0]).reshape(14,1)
control_model = Symbolic_model()
pee_ref = control_model.forward_kinemetics(x_ref[0:7])

for i in random_index:
    y = reshape_output[i,:,:].reshape(-1,17)
    plot_measurements(t, y, pee_ref)
    action = out_action[i,:,:].reshape(-1,7)
    u_lbls = [fr'$u_{k}$ [N*m]' for k in range(1, 8)]
    _, axs_u = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    for k, ax in enumerate(axs_u.reshape(-1)):
        ax.step(t[:-1], action[:, k])
        ax.set_ylabel(u_lbls[k])
        ax.set_xlabel('time')
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    
"""

# how to draw the torque

# for i in range(action.shape[0]):

"""
x_ref = np.array([-0.44,0.14,2.02,-1.61,0.57,-0.16,-1.37,0,0,0,0,0,0,0]).reshape(14,1)
control_model = Symbolic_model()
pee_ref = control_model.forward_kinemetics(x_ref[0:7])
u_ref = control_model.gravity_torque(x_ref[0:7])

t = np.arange(0, 200+1) * 0.01
y = np.load('mpc_result/output_single_point_with_wall_y.npy')
plot_measurements(t, y, pee_ref)
u = np.load('mpc_result/torque_single_point_with_wall_y.npy')
plot_torque(t,u)
print(y.shape)
"""

"""
y = np.load('mpc_result/circle_output_final.npy') 
print(y.shape)
t = np.arange(0, 200) * 0.02
circle= np.loadtxt('mpc_result/200circle_xy0.1_ee.txt',delimiter=',')
# plot_circle_xyz(t,y,circle)
plot_circle_3d(y)
"""
"""
val = np.load('expert_data/validation_states.npy').reshape(60,-1,14)
print(val.shape)
val_act = np.load('expert_data/validation_actions.npy').reshape(60,-1,7)
tet = np.load('expert_data/11state_0.3.npy')
tet_act = np.load('expert_data/11action_0.3.npy')
tet = np.delete(tet,40,axis=1)
print(tet.shape)
train_states = np.load('expert_data/train_states.npy').reshape(5940,-1,14)
train_actions = np.load('expert_data/train_actions.npy').reshape(5940,-1,7)
index = random.sample(range(0, 5940),1430)


new_state = val
new_state = np.vstack((new_state,tet))
new_state = np.vstack((new_state,train_states[index,:,:]))
print(new_state.shape)

new_action = val_act
new_action = np.vstack((new_action,tet_act))
new_action = np.vstack((new_action,train_actions[index,:,:]))
print(new_action.shape)
np.save('expert_data/test_actions.npy',new_action)
np.save('expert_data/test_states.npy',new_state)
"""

bc = np.load('training_logger/bc_reward.npy')
dagger = np.load('training_logger/dagger_reward.npy')
x_values = range(len(bc))
plt.plot(x_values, bc, marker='o')
plt.ylim(-5, 0)
plt.xlabel('test examples')
plt.ylabel('reward')
plt.title('Behavior Cloning Reward')
plt.show()

plt.plot(x_values, dagger, marker='o',color='orange')
plt.ylim(-5, 0)
plt.xlabel('test examples')
plt.ylabel('reward')
plt.title('Dagger Reward')
plt.show()