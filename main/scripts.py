import numpy as np
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


from model.iiwa14_model import Symbolic_model

from utils.plotting import plot_circle_3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.plotting import plot_circle_xyz,plot_torque,plot_q_dq
from utils.plotting import *
# state = np.load('expert_data/train_states.npy')
action = np.load('expert_data/2output_0.3.npy')
print(action.shape)

t = np.arange(0, 40 +1) * 0.01
x_ref = np.array([-0.44,0.14,2.02,-1.61,0.57,-0.16,-1.37,0,0,0,0,0,0,0]).reshape(14,1)
control_model = Symbolic_model()
pee_ref = control_model.forward_kinemetics(x_ref[0:7])

for i in range(action.shape[0]):
    y = action[i,:,:].reshape(-1,17)
    plot_measurements(t, y, pee_ref)
    
"""
# how to draw the torque
step = 40
dt = 0.05
time = np.arange(0,(step+1)*dt,dt)
for i in range(action.shape[0]):
    u_lbls = [fr'$u_{k}$ [N*m]' for k in range(1, 8)]
    _, axs_u = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    for k, ax in enumerate(axs_u.reshape(-1)):
        ax.step(time[:-1], action[i,:, k])
        ax.set_ylabel(u_lbls[k])
        ax.set_xlabel('time')
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
"""

   
