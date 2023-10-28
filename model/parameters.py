import os
import numpy as np
import casadi as cs
from acados_template import AcadosModel

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# path to a folder with the model urdf file
model_folder = 'model/urdf'
# model_folder = 'urdf'
urdf_file = 'iiwa14.urdf'
urdf_path = os.path.join(model_folder,urdf_file)

rnea = cs.Function.load(os.path.join(model_folder, 'rnea.casadi'))

# create symbolic variables for joint positions, velocities and toques
q = cs.MX.sym('q', 7)
dq = cs.MX.sym('dq', 7)
u = cs.MX.sym('u', 7)
z = cs.MX.sym('z', 3) 
x = cs.vertcat(q, dq)
xdot = cs.MX.sym('xdot', 3)
        
dq_ddq = np.zeros(7)
g = rnea(q,dq_ddq,dq_ddq)
print(g)