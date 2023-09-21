"""
Implements a gym environment for the robot arm. 
The environment is used for simulation and imitation learning.
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
from typing import Tuple
import gym
from gym import spaces
from model.iiwa14_model import Symbolic_model
from simulation.simulation import Simulator, SimulatorOptions

class iiwa14EnvOptions:
  
    def __init__(self,dt:float,x_start: np.ndarray,x_end: np.ndarray,
                 sim_time: float,sim_noise_R: np.ndarray,contr_input_state: str) -> None:
        
        if dt is None:
            self.dt = 0.01
        else:
            self.dt = dt 
        # x_start is used to set the initial state of the robot, if None a random state is sampled in reset()
        if x_start is None:
             self.x_start = None
        else:
            self.x_start = x_start
            
        if x_end is None:
            self.x_end: np.ndarray = np.array([1.8,0.5,-0.2,-0.55,1.25,-1.2,0.25,0,0,0,0,0,0,0]).reshape(14,1)
        else:
            self.x_end = x_end
      
        if sim_time is None:
            self.sim_time = 2 
        else:
            self.sim_time = sim_time
            
        if sim_noise_R is None:
            self.sim_noise_R = None
        else:
            self.sim_noise_R = sim_noise_R
            
        if contr_input_state is None:
            self.contr_input_state = 'real'
        else:
            self.contr_input_state = contr_input_state
            
        self.render_mode = None
        self.maximum_torques: np.ndarray = np.array([320,320,176,176,110,40,40])
        self.goal_dist_euclid: float = 0.01
        self.goal_min_time: float = 1 


class iiwa14Env(gym.Env):
    
    def __init__(self,options: iiwa14EnvOptions) -> None:
        self.options = options
        self.sim_model = Symbolic_model()
        self.dt = self.options.dt
        self._state = self.options.x_start
        self.x_final= self.options.x_end
        self.pee_final = self.sim_model.forward_kinemetics(self.x_final[0:7])
        self.max_intg_steps = int(self.options.sim_time/self.options.dt)
        self.no_intg_steps = 0
        
        # define simulator
        sim_opts = SimulatorOptions(
            dt = self.dt,
            n_iter = self.max_intg_steps,
            R = self.options.sim_noise_R,
            contr_input_state = self.options.contr_input_state
        )
        self.simulator = Simulator(self.sim_model, controller=None, integrator='cvodes', opts = sim_opts)
        
        # define action and observation space
        nx_ = self.sim_model.nx
        self.observation_space = spaces.Box(np.array([-np.pi * 200] * nx_),np.array([np.pi * 200] * nx_),dtype=np.float64)
        self.action_space = spaces.Box(-options.maximum_torques, options.maximum_torques,dtype=np.float64)
        
        self.render_mode = self.options.render_mode
        self.goal_dist_counter = 0
        self.stop_if_goal_reached = True
        
    def reset(self):
        if self._state is None:
            self._state = self.sample_rand_config()
        
        self.simulator.reset(x0=self._state)
        self.no_intg_steps = 0
        
        return self._state
    
    def step(self,a) -> Tuple[np.ndarray, float, bool, dict]:
        
        self._state = self.simulator.step(a)
        
        self.no_intg_steps += 1
        
        # define reward as Euclidian distance to goal
        pee_current = self.sim_model.pee(self._state[:int(self.sim_model.nq)]) 
        dist = np.linalg.norm(pee_current - self.pee_final,2)  
        reward = -dist * self.options.dt
        
        # check if goal is reached
        done = bool(self.terminal(dist))
        
        observation = self._state[:,0]
        
        info = {}
        return(observation, reward, done, info)
    
    def terminal(self,dist:float):
        if dist < self.options.goal_dist_euclid:
            self.goal_dist_counter += 1
        else:
            self.goal_dist_counter = 0
        
        done = False
        
        if (self.goal_dist_counter >= self.options.goal_min_time/self.options.dt) and self.stop_if_goal_reached:
            done = True
        if self.no_intg_steps >= self.max_intg_steps:
            done = True
        return bool(done)
    
    def sample_rand_config(self):  
        q = []
        q_range = np.deg2rad([170,120,170,120,170,120,175])
        alpha = 0.3
        for limit in q_range:
            single_joint = np.random.uniform(alpha * limit, -alpha * limit)
            q.append(single_joint)
        q = np.array(q).reshape(7,1)
        dq = np.zeros((7,1))
        x = np.vstack((q,dq))
        return x  
    
if __name__ == "__main__":
    env_options = iiwa14EnvOptions(dt=0.01,x_start=None,x_end=None,
                                   sim_time=3,sim_noise_R=None,contr_input_state=None)
    env = iiwa14Env(env_options)
    x_start = env.reset()
    print(x_start)
    next_observation,reward,done,_ = env.step(np.array([25,25,25,25,25,25,25]))
    print(next_observation)
    print(reward)
    print(done)