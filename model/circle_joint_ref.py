from __future__ import print_function

import numpy as np
from numpy.linalg import norm, solve

import pinocchio
import os

""" 
this file is used to save the inverse kinematics of the circle point
it must be run in anaconda environment installed with pinocchio
"""

model_folder = 'model/urdf/'
urdf_file = 'iiwa14.urdf'
urdf_path = os.path.join(model_folder,urdf_file)
model = pinocchio.buildModelFromUrdf(urdf_path)
  = model.create()

pee_ref = np.loadtxt('/200circle_xy0.1.txt',delimiter=',')
pee_vel_ref = np.loadtxt('/200circle_xy0.1_vel.txt',delimiter=',')
q = np.array([[0, -1.58, 0, -1.58, 0, 0, 0]])
q_ref = np.repeat(q,pee_ref.shape[0],axis=0)
joint_velocities = np.zeros((pee_ref.shape[0],7))
JOINT_ID= model.getJointId('iiwa14_joint_7') 
eps    = 1e-4
IT_MAX = 1000
DT     = 1e-1
damp   = 1e-6

for j in range(pee_ref.shape[0]):
    oMdes = pinocchio.SE3(np.eye(3), pee_ref[j,:])
    q = pinocchio.neutral(model)
    i = 0 
    while True:
        pinocchio.forwardKinematics(model,,q)
        # the difference between the current position and the desired position
        dMi = oMdes.actInv(.oMi[JOINT_ID]) 
        # gives the difference between the current and desired poses in a vector form
        err = pinocchio.log(dMi).vector 
        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pinocchio.computeJointJacobian(model,,q,JOINT_ID)
        v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pinocchio.integrate(model,q,v*DT)
        q_2pi = (np.array(q) + np.pi) % (2 * np.pi) - np.pi
        # if not i % 10:
            # print('%d: error = %s' % (i, err.T))
        i += 1

    if success:
        print("Convergence achieved!")
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
    q_ref[j,:] = q_2pi.flatten()
    # print('\nresult: %s' % q_2pi.flatten().tolist())
    # print('\nfinal error: %s' % err.T)
    
    
    # calculate the joint velocities
    J = pinocchio.computeJointJacobian(model,,q_ref[j,:],JOINT_ID)
    pseudo_inverse_J = np.linalg.pinv(J)
    joint_velocities[j,:] = np.dot(pseudo_inverse_J,pee_vel_ref[j,:])

# print(q_ref)
np.savetxt('/200circle_joint_xy0.1.txt', q_ref, delimiter=',', fmt='%f')
np.savetxt('/200circle_joint_xy0.1_vel.txt', joint_velocities, delimiter=',', fmt='%f')
