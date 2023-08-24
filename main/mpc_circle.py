import numpy as np
import time
import os
import casadi as cs

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from model.iiwa14_model import Symbolic_model
from mpc_controller.mpc import MPC, MpcOptions
from environment.gym_env import iiwa14EnvOptions, iiwa14Env
from utils.plotting import plot_measurements
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


DT = 0.02 # sampling time
N_mpc = 2 # prediction horizon
N_ITER = 1 #number of simulation iterations

Q_0 = np.zeros(7)
Q_ref = q = np.loadtxt('/200circle_joint_xy0.1.txt',delimiter=',')
DQ_0 = np.zeros(7)
# DQ_ref = np.zeros(7)
DQ_ref = np.loadtxt('/200circle_joint_xy0.1_vel.txt',delimiter=',')
pee_ref = np.loadtxt('/200circle_xy0.1.txt',delimiter=',')

R_Q_sim = np.array([0] * 7)
R_DQ_sim = np.array([0] * 7)
R_PEE_sim = np.array([0] * 3) 

CNTR_INPUT_STATES = 'real'
  

if __name__ == "__main__":

    dynamic_model = Symbolic_model()

    x = []
    u = []
    y = []
    for i in range(np.shape(Q_ref)[0]-1):
        
        x_ref = np.vstack((Q_ref[i+1,:].reshape(7,1),DQ_ref[i+1,:].reshape(7,1)))
        
        if i== 0:
            q0 = [0,0.19,0,1.70,0,1.5,0]
            dq0 = np.zeros(7)
            x0 = np.hstack((q0,dq0))
            pee0 = dynamic_model.forward_kinemetics(q0)
            
        else:
            x0 = x_point[-1,:]
            pee0 = y_point[-1,14:17]
    # create environment
        env_opts = iiwa14EnvOptions(
            dt = DT,
            x_start = x0.T,
            x_end = x_ref,
            sim_time = N_ITER*DT,
            sim_noise_R = np.diag([*R_Q_sim, *R_DQ_sim, *R_PEE_sim]),
            contr_input_state = CNTR_INPUT_STATES
        )
        env = iiwa14Env(env_opts)

        #  create mpc controller
        mpc_opts = MpcOptions(tf=DT*N_mpc, n=N_mpc)
        # print(mpc_opts.get_sample_time())
        assert round(mpc_opts.get_sample_time(),4) == DT
        MPC_controller = MPC(dynamic_model, np.reshape(x0,(14,1)), pee0.reshape(3,1), mpc_opts)
    
        # provide reference for mpc controller
        # print(pee_ref)
        u_ref = dynamic_model.gravity_torque(Q_ref[i,:]) 
        MPC_controller.set_reference_point(x_ref, pee_ref[i,:].reshape(3,1), u_ref.reshape(7,1))
    
        # simulation
        nq = dynamic_model.nq
        state = env.reset()
        qk , dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
        print("Initial state: ", state)
    
        start_time = time.time()
        for j in range(env.max_intg_steps):
            a = MPC_controller.compute_torques(q=qk, dq=dqk,t = j*DT)
            state, reward, done, info = env.step(a)
            qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1 )
            if done:
                break
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time: ", execution_time, " seconds")
    
        x_point , u_point , y_point = env.simulator.x, env.simulator.u, env.simulator.y
        
        if i == 0 :
            x.append(x_point)
            y.append(y_point)   
        else:
            x.append(x_point[1:,:])
            y.append(y_point[1:,:])
            
        u.append(u_point)
        
        
        
    # save the circle_trajectoty
    x = np.vstack(x)
    u = np.vstack(u)
    y = np.vstack(y)
   
    np.save("/circle_torque_final.npy", u)  
    np.save("/circle_state_final.npy", x) 
    np.save("/circle_output_final.npy", y)   
    
    # plot the circle trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(np.shape(y)[0]):
        point = y[i,14:17]
        ax.scatter(point[0],point[1],point[2], c='b', marker='o')
        
    plt.show()
