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

model = Symbolic_model()

x = np.load("expert_data/state_0.3.npy")
u = np.load("expert_data/action_0.3.npy")

for i in range(len(u)):
    u_single = u[i,:,:].reshape(-1,7)
    t = np.arange(0, len(u_single)+1) * 0.05
    plot_torque(t,u_single)
