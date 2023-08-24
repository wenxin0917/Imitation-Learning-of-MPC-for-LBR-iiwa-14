import pinocchio as pin
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

""" this file is used to save the point of the circle, if the trajectory is changed, this file 
    should be changed as well like N and the radius of the circle, also the center of the circle
"""

def circle(c,r,a,b,theta):
    # a and b are orthogonal vectors
    assert np.dot(a.T, b).item() == 0
    out = c + r*np.cos(theta)*a + r*np.sin(theta)*b
    return out.T

def circle_velocity(c,r,a,b,theta):
    # a and b are orthogonal vectors
    assert np.dot(a.T, b).item() == 0
    velocity = -r*np.sin(theta)*a + r*np.cos(theta)*b
    return velocity.T


# design the circular trajectory for the end-effector
r = 0.1
c_ee = np.array([[4.58, 0, 0.925]]).T
a = np.array([[1., 0., 0.]]).T
b = np.array([[0., 1., 0.]]).T
s_ee = np.linspace(0., 2*np.pi, 200) 
s = np.linspace(0., 2*np.pi, 200) 
pee_ref = circle(c_ee, r, a, b, s_ee) 
np.savetxt('/200circle_xy0.1_ee.txt', pee_ref, delimiter=',', fmt='%f')

c = np.array([[4.58, 0, 0.88]]).T
joint7_ref = circle(c, r, a, b, s) 
np.savetxt('/200circle_xy0.1.txt', joint7_ref, delimiter=',', fmt='%f')

joint7_linear_vel = circle_velocity(c, r, a, b, s)
# calculate the angular velocity
joint7_angu_vel = np.zeros((len(s),3))
time_interval = 0.02*1
angu_vel = (s[1]-s[0])/time_interval
joint7_angu_vel[:,2] = angu_vel
joint7_vel = np.hstack((joint7_linear_vel, joint7_angu_vel))

np.savetxt('/200circle_xy0.1_vel.txt', joint7_vel, delimiter=',', fmt='%f') 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(200):
    point0 = pee_ref[i,:]
    point1 = joint7_ref[i,:]
    ax.scatter(point0[0],point0[1],point0[2], c='b', marker='o')
    ax.scatter(point1[0],point1[1],point1[2], c='r', marker='o')
    
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()