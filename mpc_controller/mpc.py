
from typing import TYPE_CHECKING, Tuple
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
from tempfile import mkdtemp
import scipy
# from optimal_planner import design_optimal_circular_trajectory
from scipy.interpolate import interp1d
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from model.iiwa14_model import Symbolic_model

"""
when it is used for the calculation of circle trajectory
Q_q = 1e7, Q_dq = 1e3, Q_qe = 1e7, Q_dqe = 1e3,
self.nlp_solver_max_iter : int = 100
self.P = np.diagflat(np.array([1]*3))* 3e7
self.Pn = np.diagflat(np.array([1]*3))* 1e7
"""   

"""
when it is used for the calculation of single point
Q_q = 0.1, Q_dq = 0.01, Q_qe = 5e2, Q_dqe = 5e1,
self.nlp_solver_max_iter : int = 50
self.P = np.diagflat(np.array([1]*3))* 3e3
self.Pn = np.diagflat(np.array([1]*3))* 1e4
"""

"""
when it is used for the expert data generation( DT ==0.05)
Q_q = 0.1, Q_dq = 5e1, Q_qe = 5e2, Q_dqe = 5e1,
self.nlp_solver_max_iter : int = 100
self.P = np.diagflat(np.array([1]*3))* 3e3
self.Pn = np.diagflat(np.array([1]*3))* 1e4
"""

# define penalty element of Q for state x 
Q_q = 0.1
Q_dq = 5e1
# define penalty element of Qn for terminal state x
Q_qe = 5e2
Q_dqe = 5e1

class MpcOptions:
    def __init__(self, tf: float=2, n: int=30) -> None:
        self.n : int = n
        self.tf : float = tf
        self.nlp_solver_max_iter : int = 100
        self.condensing_relative: float=1
        # define the wall constraint
        self.wall_0deg_constraint_on: bool = False
        self.wall_axis: int = 1
        self.wall_value: float = 0.4
        self.wall_pos_side: bool = False
        self.wall_30deg_constraint_on: bool = False
        # define the penalty matrices Q R and P
        self.Q = np.diagflat(np.array([[Q_q]*7 + [Q_dq]*7]))
        self.R = np.diagflat(np.array([0.001]*7))
        self.P = np.diagflat(np.array([1]*3))* 3e3
        self.Qn = np.diagflat(np.array([[Q_qe]*7 + [Q_dqe]*7]))
        self.Pn = np.diagflat(np.array([1]*3))* 1e4
        self.speed_slack: float = 1e6
        self.wall_slack: float = 1e4
        
    def get_sample_time(self) -> float:
        return self.tf/self.n
    
class MPC:
    def __init__(self, model: "Symbolic_model", 
               x0: np.ndarray = None, pee_0: np.ndarray = None,
               options: MpcOptions = MpcOptions()) -> None:
        
        # x0, pee_0 column vector 2D array
        
        # this x0 and pee_0 are only used to initialize the controller,
        if x0 is None:
            x0 = np.zeros(model.nx,1)
        if pee_0 is None:
            pee_0 = np.zeros(3,1)
        
        # initialize the MPC conditions
        self.tau_max = model.tau_max
        self.dq_max = model.dq_max
        self.dymodel = model
        mpcmodel, constraint_expr = model.get_acados_model()
        self.model = mpcmodel
        self.options = options
        self.iteration_conuter = 0
        self.debug_time = []
        
        # initialize the parameter for circle trajectory
        self.inter_t2q = None
        self.inter_t2dq = None
        self.inter_pee = None
        
        # create ocp project to formulate the OCP optimal control problem
        ocp = AcadosOcp()
        ocp.model = mpcmodel
        ocp.dims.N = options.n
        ocp.code_export_directory = mkdtemp()
        
        # OCP parameters
        nx = self.dymodel.nx
        nu = self.dymodel.nu
        nz = self.dymodel.nz
        ny = nx + nu + nz
        ny_e = nx + nz
        self.nu = nu
        self.nx = nx
        
        # dimension check
        assert (nx == options.Q.shape[0] == options.Qn.shape[0])
        assert (nu == options.R.shape[0])
        assert (nz == options.P.shape[0] == options.Pn.shape[0])
        
        # set cost module
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        ocp.cost.W = scipy.linalg.block_diag(options.Q, options.R, options.P)
        ocp.cost.W_e = scipy.linalg.block_diag(options.Qn, options.Pn)
        
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :] = np.eye(nx)
        
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:nx+nu, :] = np.eye(nu)
        
        ocp.cost.Vz = np.zeros((ny, nz))
        ocp.cost.Vz[nx+nu:, :] = np.eye(nz)
        
        ocp.cost.Vx_e = np.zeros((ny_e, nx))
        ocp.cost.Vx_e[:nx, :] = np.eye(nx)
        
        ocp.cost.Vz_e = np.zeros((ny_e, nz))
        ocp.cost.Vz_e[nx:, :] = np.eye(nz)
        
        goal_x = x0
        goal_cartesian = pee_0
        ocp.cost.yref = np.vstack((goal_x, np.zeros((nu,1)), goal_cartesian)).flatten()
        ocp.cost.yref_e = np.vstack((goal_x, goal_cartesian)).flatten()
        # the reference is only used to initialize the solver, there is a function to set the reference point
        
        # set constraints
        ocp.constraints.constr_type = 'BGH'
        ocp.constraints.x0 = x0.reshape((nx,))
        
        # define the bound for the control input u
        # idxbu: index of soft bounds on control inputs
        ocp.constraints.lbu = - self.tau_max
        ocp.constraints.ubu = self.tau_max
        ocp.constraints.idxbu = np.arange(nu)
        
        # define the bound for the state x(7,14)
        ocp.constraints.lbx = - self.dq_max
        ocp.constraints.ubx = self.dq_max
        ocp.constraints.idxbx = np.arange(7,14)
        
        ocp.constraints.lbx_e = - self.dq_max
        ocp.constraints.ubx_e = self.dq_max
        ocp.constraints.idxbx_e = np.arange(7,14)
        ocp.constraints.idxsbx = np.arange(7,14) 
        
        # set constraints for the wall
        if options.wall_0deg_constraint_on:
            ocp.model.con_h_expr = constraint_expr[options.wall_axis]
            n_wall_constraints = 1
            ns = n_wall_constraints
            nsh = n_wall_constraints
            self.current_slacks = np.zeros((ns,))
            # define the penalty matrix for the slack variable
            ocp.cost.zl = np.array([1e3]*7+[1e4]*n_wall_constraints)
            ocp.cost.Zl = np.array([options.speed_slack]*7+[options.wall_slack]*n_wall_constraints)
            ocp.cost.zu = ocp.cost.zl
            ocp.cost.Zu = ocp.cost.Zl
            # define the bounds for the inequality constraints
            ocp.constraints.lh = np.ones((n_wall_constraints,))* (options.wall_value if options.wall_pos_side else -1e3)
            ocp.constraints.uh = np.ones((n_wall_constraints,))* (options.wall_value if not options.wall_pos_side else 1e3)
            # define the bounds on slacks corresponding to soft lower and upper bounds for nonlinear inequalities
            ocp.constraints.lsh = np.zeros(nsh)
            ocp.constraints.ush = np.zeros(nsh)
            # indices of soft nonlinear constraints within the indices of nonlinear constraints
            ocp.constraints.idxsh = np.array(range(n_wall_constraints))
        elif options.wall_30deg_constraint_on:
            ocp.model.con_h_expr = 0.577* constraint_expr[0]- constraint_expr[1] - 2.485
            n_wall_constraints = 1
            ns = n_wall_constraints
            nsh = n_wall_constraints
            self.current_slacks = np.zeros((ns,))
            # define the penalty matrix for the slack variable
            ocp.cost.zl = np.array([1e3]*7+[1e4]*n_wall_constraints)
            ocp.cost.Zl = np.array([options.speed_slack]*7+[options.wall_slack]*n_wall_constraints)
            ocp.cost.zu = ocp.cost.zl
            ocp.cost.Zu = ocp.cost.Zl
            # define the bounds for the inequality constraints
            ocp.constraints.lh = np.zeros((n_wall_constraints,))
            ocp.constraints.uh = np.ones((n_wall_constraints,))*1e3
            # define the bounds on slacks corresponding to soft lower and upper bounds for nonlinear inequalities
            ocp.constraints.lsh = np.zeros(nsh)
            ocp.constraints.ush = np.zeros(nsh)
            # indices of soft nonlinear constraints within the indices of nonlinear constraints
            ocp.constraints.idxsh = np.array(range(n_wall_constraints))
            
        else:   
            ocp.cost.zl = np.array(([0]*7))
            ocp.cost.zu = np.array(([0]*7))
            ocp.cost.Zl = np.array([options.speed_slack]*7)
            ocp.cost.Zu = np.array([options.speed_slack]*7)
        
        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES # PARTIAL_CONDENSING_HPIPM
        ocp.solver_options.qp_solver_cond_N = int(options.n * options.condensing_relative)
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.nlp_solver_type = 'SQP'  # SQP_RTI, SQP
        ocp.solver_options.nlp_solver_max_iter = options.nlp_solver_max_iter

        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.qp_solver_cond_N = options.n

        # set prediction horizon
        ocp.solver_options.tf = options.tf

        self.acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_' + 'iiwa14' + '.json')
        # the reason for creating an OCP before creating the solver is to seperate the problem formulation from the numerical optimization process
    
    def reset(self) -> None:
        self.debug_time = []
        self.iteration_conuter = 0
    
    def set_reference_point(self, x_ref: np.ndarray, pee_ref: np.ndarray, u_ref: np.array) -> None:
        yref = np.vstack((x_ref, u_ref, pee_ref)).flatten()
        yref_e = np.vstack((x_ref, pee_ref)).flatten()
        for stage in range(self.options.n):
            self.acados_ocp_solver.cost_set(stage, "yref", yref) 
        self.acados_ocp_solver.cost_set(self.options.n, "yref", yref_e)
        
    def compute_torques(self, q: np.ndarray, dq: np.ndarray, t:float = None) -> np.ndarray:
        
        # set initial state
        xcurrent = np.vstack((q, dq))
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        # solve the OCP
        status = self.acados_ocp_solver.solve()
        
        self.debug_time.append(self.acados_ocp_solver.get_stats("time_tot")[0])
        if status != 0 and status != 2:
            u = np.zeros((self.nu,))
            print("no optimal solution")
            # raise RuntimeError('acados returned status {} in time step {}.'.format(status, self.iteration_conuter))
            
        else:  
            # get solution
            u = self.acados_ocp_solver.get(0, "u")
            
        self.iteration_conuter += 1
        
        return u
        
        