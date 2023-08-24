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

x = np.load("/circle_state_final.npy")
u = np.load("/circle_torque_final.npy")
circle = np.loadtxt("/200circle_xy0.1_ee.txt",delimiter=',')
y = np.load("/circle_output_final.npy")
t = np.arange(0, len(x)) * 0.02


plot_circle_xyz(t,y,circle)
plot_torque(t,u)
plot_q_dq(t,y)