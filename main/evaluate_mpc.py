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


DT = 0.01 # sampling time
N_mpc = 30 # prediction horizon
N_ITER = 200 #number of simulation iterations

Q_0 = np.zeros(7)
Q_ref = np.array([-0.44,0.14,2.02,-1.61,0.57,-0.16,-1.37])
DQ_0 = np.zeros(7)
DQ_ref = np.zeros(7)

R_Q_sim = np.array([0] * 7)
R_DQ_sim = np.array([0] * 7)
R_PEE_sim = np.array([0] * 3) 

CNTR_INPUT_STATES = 'real'
  

if __name__ == "__main__":

    # create environment
    env_opts = iiwa14EnvOptions(
        dt = DT,
        x_start = None,
        x_end = Q_ref,
        sim_time = N_ITER*DT,
        sim_noise_R = np.diag([*R_Q_sim, *R_DQ_sim, *R_PEE_sim]),
        contr_input_state = CNTR_INPUT_STATES
    )
    env = iiwa14Env(env_opts)

    #  create mpc controller
    dynamic_model = Symbolic_model()
    mpc_opts = MpcOptions(tf=DT*N_mpc, n=N_mpc)
    assert mpc_opts.get_sample_time() == DT
    
    x0_mpc = np.hstack((Q_0,DQ_0))
    pee_0 = dynamic_model.forward_kinemetics(Q_0)
    MPC_controller = MPC(dynamic_model, np.reshape(x0_mpc,(14,1)), pee_0, mpc_opts)
    
    # provide reference for mpc controller
    q_ref = Q_ref.reshape(7,1)
    dq_ref = DQ_ref.reshape(7,1)
    x_ref = np.vstack((q_ref,dq_ref))
    pee_ref = dynamic_model.forward_kinemetics(q_ref)
    # print(pee_ref)
    u_ref = dynamic_model.gravity_torque(Q_ref) 
    MPC_controller.set_reference_point(x_ref, pee_ref, u_ref)
    
    # simulation
    nq = dynamic_model.nq
    state = env.reset()
    qk , dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
    print("Initial state: ", state)
    
    start_time = time.time()
    for i in range(env.max_intg_steps):
        a = MPC_controller.compute_torques(q=qk, dq=dqk,t = i*DT)
        state, reward, done, info = env.step(a)
        qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1 )
        if done:
            break
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ", execution_time, " seconds")
    
    x , u , y = env.simulator.x, env.simulator.u, env.simulator.y
    
    # save the trajectoty with wall 0.577x - 2.485 -y >0
    np.save("/torque_single_point_with_wall_30.npy", u)  
    np.save("/state_single_point_with_wall_30.npy", x)    
    np.save("/output_single_point_with_wall_30.npy", y)
    
    # save the trajectoty with wall y< 0.4
    # np.save("/torque_single_point_with_wall_y.npy", u)
    # np.save("/state_single_point_with_wall_y.npy", x)
    #np.save("/output_single_point_with_wall_y.npy", y)
    
    # save the trajectoty without wall
    # np.save("/torque_single_point_without_wall.npy", u)  
    # np.save("/state_single_point_without_wall.npy", x) 
    # np.save("/output_single_point_without_wall.npy", y))    
    t = np.arange(0, env.max_intg_steps +1) * env.dt
    plot_measurements(t, y, pee_ref)
