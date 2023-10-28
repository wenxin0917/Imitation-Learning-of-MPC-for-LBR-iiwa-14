import casadi as cs
import numpy as np
# from numpy.random import multivariate_normal

# import sys
# import os
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)

from model.iiwa14_model import Symbolic_model

# Measurement noise covariance matrix
R_Q = [3e-6] * 7
R_DQ = [2e-3] * 7
R_PEE = [1e-4] * 3


class SimulatorOptions:
    def __init__(self,dt,n_iter,R,contr_input_state) -> None:
        
        self.rtol: float = 1e-8
        self.atol: float = 1e-10
        if R is None:
            self.R = np.diag([*R_Q, *R_DQ, *R_PEE])
        else:
            self.R = R
            
        if contr_input_state is None:
            self.contr_input_state = 'real'
        else:
            self.contr_input_state = contr_input_state
        
        if dt is None:
            self.dt = 0.05
        else:
            self.dt = dt

        if n_iter is None:
            self.n_iter: int = 100
        else:
            self.n_iter = n_iter
    
class Simulator:
    def __init__(self, robot, controller, integrator,
                 opts: SimulatorOptions ) -> None:
        # check the model
        if integrator in ['collocation','cvodes']:
            assert isinstance(robot, Symbolic_model)
            
        # initialize the simulator
        self.robot = robot
        self.controller = controller
        self.integrator = integrator
        self.opts = opts
        
        # create an integrator
        if self.integrator in ['collocation', 'cvodes']:
            dae = {'x': self.robot.x, 'p': self.robot.u, 'ode': self.robot.rhs}
            if self.integrator == 'collocation':
                opts = {'t0': 0, 'tf': self.opts.dt, 'number_of_finite_elements': 3,
                        'simplify': True, 'collocation_scheme': 'radau',
                        'rootfinder': 'fast_newton', 'expand': True,
                        'interpolation_order': 3} 
            else:
                opts = {'t0': 0, 'tf': self.opts.dt, 'abstol': self.opts.atol, 'reltol': self.opts.rtol,
                        'nonlinear_solver_iteration': 'newton', 'expand': True,
                        'linear_multistep_method': 'bdf'}
            I = cs.integrator('I', self.integrator, dae, opts)
            x_next = I(x0=self.robot.x, p=self.robot.u)["xf"]
            self.F = cs.Function('F', [self.robot.x, self.robot.u], [x_next])
            
        self.x = None
        self.u = None
        self.y = None
        self.k = 0
        
    def reset(self,x0) -> np.ndarray:
        # x0 dimension (14,1)
        self.k = 0
        
        self.x = np.zeros((self.opts.n_iter+1, self.robot.nx))
        self.u = np.zeros((self.opts.n_iter, self.robot.nu))
        self.y = np.zeros((self.opts.n_iter+1, self.robot.ny))
        
        self.x[0,:]=x0.reshape(14,)
        self.y[0,:]=self.robot.output(self.x[0,:].T).flatten()
        # self.y[0,:]=self.robot.output(self.x[0,:].T).flatten() + multivariate_normal(np.zeros(self.robot.ny), self.opts.R)

        return self.x[0,:]
    
    def ode_wrapper(t,x,robot,tau):
        
        return robot.odefun(x,tau)
    
    def integrator_step(self,x,u) -> np.ndarray:
        # x 1D , u 1D , dt step size of the integrator
        if self.integrator in ['collocation', 'cvodes']:
            x_next = np.array(self.F(x, u)).flatten()
        else:
            raise ValueError
        return x_next
    
    def step(self,input_tau: np.ndarray) -> np.ndarray:
        self.u[[self.k], :] = input_tau
        x_next = self.integrator_step(self.x[[self.k], :], self.u[[self.k], :])
        self.x[self.k + 1, :] = x_next
        self.y[self.k + 1, :] = self.robot.output(self.x[[self.k + 1], :].T).flatten() 
        # self.y[self.k + 1, :] = (self.robot.output(self.x[[self.k + 1], :].T).flatten() +
        #                         multivariate_normal(np.zeros(self.robot.ny), self.opts.R))
        self.k += 1
        return self.x[[self.k], :].T
    
    
    def mix_step(self,expert_tau,policy_tau,mixture_ratio) -> np.ndarray:
        self.u[[self.k], :] = (1-mixture_ratio)* expert_tau + mixture_ratio*policy_tau
        x_next = self.integrator_step(self.x[[self.k], :], self.u[[self.k], :])
        self.x[self.k + 1, :] = x_next
        self.y[self.k + 1, :] = self.robot.output(self.x[[self.k + 1], :].T).flatten() 
        self.k += 1
        return self.x[[self.k], :].T
    
    
    def simulate(self, x0, n_iter: int = None):
        state = self.reset(x0, n_iter)
        nq = self.robot.nq
        qk, dqk = state[0:nq, :], state[nq:, :]
        for k in range(self.opts.n_iter):
            tau = self.controller.compute_torques(qk, dqk, t=self.opts.dt * k,
                                                  y=self.y[k, :].T)
            state = self.step(input_tau=tau)
            qk, dqk = state[0:nq, :], state[nq:, :]

        return self.x, self.u, self.y
    
if __name__ == "__main__":
    
    model = Symbolic_model()
    options = SimulatorOptions(dt=0.01,n_iter=100,R=None,contr_input_state=None)
    sim = Simulator(model,None,'cvodes',options)
    x0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(14,1)
    x_reset = sim.reset(x0)
    print(x_reset)
    x_next = sim.step(np.array([25,25,25,25,25,25,25]))
    print(x_next)