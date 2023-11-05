import sys
import os
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from model.iiwa14_model import Symbolic_model
from mpc_controller.mpc import MPC, MpcOptions
from environment.gym_env import iiwa14EnvOptions, iiwa14Env
from utils.plotting import *




def generate_expert_data(horizon_n,x_0,x_ref):
    """
    Generate expert data for imitation learning(BC).
    """ 
    
    # simulation parameters
    DT = 0.05 # sampling time   
    N_ITER = 40 #number of simulation iterations

    R_Q_sim = np.array([0] * 7)
    R_DQ_sim = np.array([0] * 7)
    R_PEE_sim = np.array([0] * 3) 

    CNTR_INPUT_STATES = 'real'

    # create environment
    env_opts = iiwa14EnvOptions(
        dt = DT,
        x_start = x_0,
        x_end = x_ref,
        sim_time = N_ITER * DT,
        sim_noise_R = np.diag([*R_Q_sim, *R_DQ_sim, *R_PEE_sim]),
        contr_input_state = CNTR_INPUT_STATES
    )
    env = iiwa14Env(env_opts)

    # controller
    control_model = Symbolic_model()
    mpc_opts = MpcOptions(tf=DT*horizon_n, n=horizon_n)
    pee_0 = control_model.forward_kinemetics(x_0[0:7])
    MPC_controller = MPC(control_model,x_0, pee_0, mpc_opts)
    # assert mpc_opts.get_sample_time() == DT
    
    # provide reference for mpc controller
    pee_ref = control_model.forward_kinemetics(x_ref[0:7])
    u_ref = control_model.gravity_torque(x_ref[0:7]) 
    MPC_controller.set_reference_point(x_ref, pee_ref, u_ref)
    
    # simulation
    nq = control_model.nq
    state = env.reset()
    qk , dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
    
    # start_time = time.time()
    for i in range(env.max_intg_steps):
        a = MPC_controller.compute_torques(q=qk, dq=dqk,t = i*DT)
        state, reward, done, info = env.step(a)
        qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1 )
        if done:
            break
    # end_time = time.time()
    # execution_time = end_time - start_time
    x , u , y = env.simulator.x, env.simulator.u, env.simulator.y       
    # t = np.arange(0, env.max_intg_steps +1) * env.dt
    
    # plot_measurements(t, y, pee_ref)
    # print("Execution time: ", execution_time, " seconds")
    return x,u,y

def sample_rand_initial_position(alpha:float,q_range:np.array):
        """
        Generate random initial position for the robot.
        """ 
        q = []
        for limit in q_range:
            single_joint = np.random.uniform(alpha * limit, -alpha * limit)
            q.append(single_joint)
        q = np.array(q).reshape(7,1)
        dq = np.zeros((7,1))
        x = np.vstack((q,dq))
        return x
    
def generate_expert_data_with_NN(mixture_ratio,agent,x_0):
    """
    Mix policy( Dagger ).
    """
    # MPC parameters
    DT = 0.05 # sampling time
    N_ITER = 40
    horizon_n = 5 # the same as the parameter in generate_expert_data
    x_ref = np.array([-0.44,0.14,2.02,-1.61,0.57,-0.16,-1.37,0,0,0,0,0,0,0]).reshape(14,1)
    x_0 = x_0.reshape(14,1)
    R_Q_sim = np.array([0] * 7)
    R_DQ_sim = np.array([0] * 7)
    R_PEE_sim = np.array([0] * 3) 

    CNTR_INPUT_STATES = 'real'

    # create environment
    env_opts = iiwa14EnvOptions(
        dt = DT,
        x_start = x_0,
        x_end = x_ref,
        sim_time = N_ITER * DT,
        sim_noise_R = np.diag([*R_Q_sim, *R_DQ_sim, *R_PEE_sim]),
        contr_input_state = CNTR_INPUT_STATES
    )
    env = iiwa14Env(env_opts)
    
    # controller
    control_model = Symbolic_model()
    mpc_opts = MpcOptions(tf=DT*horizon_n, n=horizon_n)
    pee_0 = control_model.forward_kinemetics(x_0[0:7])
    
    # assert mpc_opts.get_sample_time() == DT
    
    # provide reference for mpc controller
    pee_ref = control_model.forward_kinemetics(x_ref[0:7])
    u_ref = control_model.gravity_torque(x_ref[0:7]) 
    MPC_controller = MPC(control_model,x_0, pee_0, mpc_opts)
    MPC_controller.set_reference_point(x_ref, pee_ref, u_ref)
    
    # simulation
    nq = control_model.nq
    state = env.reset()
    qk , dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1)
    
    # start_time = time.time()
    for i in range(env.max_intg_steps):
        a = MPC_controller.compute_torques(q=qk, dq=dqk,t = i*DT)
        print("the torque is ",a)
        # consider the case that there is no solution for some states"""
        if np.all(a == np.zeros((7,))):
            return a,a,a
        state, reward, done, info = env.step_mix_with_policy(a,mixture_ratio,agent)
        qk, dqk = np.expand_dims(state[0:nq], 1), np.expand_dims(state[nq:], 1 )
        if done:
            break
    print("succeed collecting new states")
    x_mpc = env.simulator.x
    # print(x_mpc.shape)
    x_mpc = np.delete(x_mpc,40,axis=0)
    x , u , y = x_mpc, env.simulator.u, env.simulator.y       
    return x,u,y

    


if __name__ == "__main__":
    
    
    # generate expert data for imitation learning
    expert_data_state = np.zeros((6000,41,14))
    expert_data_action = np.zeros((6000,40,7))
    expert_data_output= np.zeros((6000,41,17))
    q_range = np.deg2rad([170,120,170,120,170,120,175])
    alpha = 0.3
    np.random.seed(10)
    used_initial_position = set()
    for i in range(6000):
        x_0 = sample_rand_initial_position(alpha,q_range)
        # in case there are repeated initial positions
        x_0_tuple = tuple(x_0.flatten())
        if x_0_tuple not in used_initial_position:
            used_initial_position.add(x_0_tuple)
        else:
            i= i-1
    used_initial_position_list = list(used_initial_position)
    
    for i in range(6000):
        x_ref = np.array([-0.44,0.14,2.02,-1.61,0.57,-0.16,-1.37,0,0,0,0,0,0,0]).reshape(14,1)
        x,u,y = generate_expert_data(5,np.array(used_initial_position_list[i]).reshape(14,1),x_ref)
        expert_data_state[i,:,:] = x
        expert_data_action[i,:,:] = u
        expert_data_output[i,:,:] = y
        print("",i)
    
    np.save('expert_data/6000expert_state_0.3.npy',expert_data_state)
    np.save('expert_data/6000expert_action_0.3.npy',expert_data_action)
    np.save('expert_data/6000expert_output_0.3.npy',expert_data_output)
