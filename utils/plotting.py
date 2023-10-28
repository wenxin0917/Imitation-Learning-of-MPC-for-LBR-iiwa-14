
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_measurements(t: np.ndarray, y: np.ndarray, pee_ref: np.ndarray = None, axs=None):
    # y output (1,17)
    # Parse measurements
    q = y[:, :7]
    dq = y[:, 7:14]
    pee = y[:, 14:]
    do_plot = True
    q_ref = [-0.44,0.14,2.02,-1.61,0.57,-0.16,-1.37]
    
    q_lbls = [fr'$q_{k}$ [rad]' for k in range(1, 8)]
    
    if axs is None:
        _, axs_q = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    else:
        axs_q = axs
        do_plot = False
    for k, ax in enumerate(axs_q.reshape(-1)):
        ax.plot(t, q[:, k])
        ax.axhline(q_ref[k], ls='--', color='red')
        ax.set_ylabel(q_lbls[k])
        ax.grid(alpha=0.5)
    axs_q[6].set_xlabel('t [s]')
    plt.tight_layout()

    dq_lbls = [fr'$\dot q_{k}$ [rad/s]' for k in range(1, 8)]
    _, axs_dq = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    for k, ax in enumerate(axs_dq.reshape(-1)):
        ax.plot(t, dq[:, k])
        ax.set_ylabel(dq_lbls[k])
        ax.grid(alpha=0.5)
    axs_dq[6].set_xlabel('t [s]')
    plt.tight_layout()
    
    
    pee_lbls = [r'$p_{ee,x}$ [m]', r'$p_{ee,y}$ [m]', r'$p_{ee,z}$ [m]']
    _, axs_pee = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
    for k, ax in enumerate(axs_pee.reshape(-1)):
        ax.plot(t, pee[:, k])

        ax.axhline(pee_ref[k], ls='--', color='red')

        ax.set_ylabel(pee_lbls[k])
        
        ax.grid(alpha=0.5)
    axs_pee[2].set_xlabel('t [s]') 
    plt.tight_layout()
    if do_plot:
        plt.show()
    
def plot_circle_3d(outputy):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(np.shape(outputy)[0]):
        point = outputy[i,14:]
        ax.scatter(point[0],point[1],point[2], c='b', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_circle_xyz(t,y,circle):

    plt.figure(figsize=(10, 5))
    # Plot x values
    plt.subplot(131)
    plt.plot(t,circle[:,0], label='reference_x')
    plt.plot(t,y[:,-3], label='x')
    plt.xlabel('time')
    plt.ylabel('X values')
    plt.legend()  

    # Plot y values
    plt.subplot(132)
    plt.plot(t,circle[:,1], label='reference_y')
    plt.plot(t,y[:,-2], label='y')
    plt.xlabel('time')
    plt.ylabel('Y values')
    plt.legend()  

    # Plot z values
    plt.subplot(133)
    plt.plot(t,circle[:,2], label='reference_z')
    plt.plot(t,y[:,-1], label='z')
    plt.xlabel('time')
    plt.ylabel('Z values')
    plt.legend()  

    plt.show()

def plot_torque(t,u):
    u_lbls = [fr'$u_{k}$ [N*m]' for k in range(1, 8)]
    _, axs_u = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    for k, ax in enumerate(axs_u.reshape(-1)):
        ax.step(t[:-1], u[:, k])
        ax.set_ylabel(u_lbls[k])
        ax.set_xlabel('time')
        ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    
def plot_q_dq(t,y):
    q = y[:, :7]
    dq = y[:, 7:14]
    
    q_lbls = [fr'$q_{k}$ [rad]' for k in range(1, 8)]
    _, axs_q = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    for k, ax in enumerate(axs_q.reshape(-1)):
        ax.plot(t, q[:, k])
        ax.set_ylabel(q_lbls[k])
        ax.grid(alpha=0.5)
    axs_q[6].set_xlabel('t [s]')
    plt.tight_layout()
    plt.show()

    dq_lbls = [fr'$\dot q_{k}$ [rad/s]' for k in range(1, 8)]
    _, axs_dq = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    for k, ax in enumerate(axs_dq.reshape(-1)):
        ax.plot(t, dq[:, k])
        ax.set_ylabel(dq_lbls[k])
        ax.grid(alpha=0.5)
    axs_dq[6].set_xlabel('t [s]')
    plt.tight_layout()
    plt.show()