
import os
import numpy as np
import casadi as cs
from acados_template import AcadosModel

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Symbolic_model:
    def __init__(self,integrator:str = 'collocation', dt: float=0.001) -> None:
        # sanity check
        assert (integrator in ['collocation','idas','cvodes'])
        
        # path to a folder with the model urdf file
        model_folder = 'model/urdf'
        # model_folder = 'urdf'
        urdf_file = 'iiwa14.urdf'
        self.urdf_path = os.path.join(model_folder,urdf_file)
        
        # dimensionality paremeters
        # nx: state dimension, nz: end-effector position dimension,ny: output dimension (q,dq,pee)
        self.nq =7  
        self.nx = 14
        self.nu = 7
        self.nz = 3 
        self.ny = 17 
        
        # velocity and torque limits
        self.tau_max = np.array([320,320,176,176,110,40,40])
        dq_max_degree = [85,85,100,75,130,135,135]
        self.dq_max = np.deg2rad(dq_max_degree)
        
        # load the forward dynamics algorithm function ABA and RNEA
        # ABA is used to calculate the acceleration given the current state and torque
        # RNEA is used to calculate the gravity torque given the current state
        self.aba = cs.Function.load(os.path.join(model_folder, 'aba.casadi'))
        self.rnea = cs.Function.load(os.path.join(model_folder, 'rnea.casadi'))
        
        # get function for the forward kinematics and velocities
        self.pee = cs.Function.load(os.path.join(model_folder, 'fkp.casadi'))
        self.vee = cs.Function.load(os.path.join(model_folder, 'fkv.casadi'))
        
        # create symbolic variables for joint positions, velocities and toques
        q = cs.MX.sym('q', self.nq)
        dq = cs.MX.sym('dq', self.nq)
        u = cs.MX.sym('u', self.nu)
        z = cs.MX.sym('z', self.nz) 
        x = cs.vertcat(q, dq)
        xdot = cs.MX.sym('xdot', self.nx)
        
        # Acceleration and right hand side of the state function
        ddq = self.aba(q,dq,u)
        rhs_state = cs.vertcat(dq, ddq)
        output_y = cs.vertcat(q,dq,self.pee(q))
        
        # calculate the derivatives of rhs_ODE f(x,u) and output y=h(x)
        drhs_state_dx = cs.jacobian(rhs_state, x)
        drhs_state_du = cs.jacobian(rhs_state, u)
        doutput_y_dx = cs.jacobian(output_y, x)
        
        # initial symbolic state parameters
        self.rhs_impl = self.pee(q)
        self.x = x
        self.xdot = xdot
        self.u = u
        self.z = z  
        self.rhs= rhs_state
        self.odefun = cs.Function('odefun', [self.x, self.u], [self.rhs], ['x', 'u'], ['dx'])
        self.outy = cs.Function('outy', [self.x], [output_y], ['x'], ['y'])
        self.df_dx = cs.Function('df_dx', [self.x, self.u], [drhs_state_dx], ['x', 'u'], ['df_dx'])
        self.df_du = cs.Function('df_du', [self.x, self.u], [drhs_state_du], ['x', 'u'], ['df_du'])
        self.douty_dx = cs.Function('douty_dx', [self.x], [doutput_y_dx], ['x'], ['douty_dx'])
        
        # create an integrator
        dae = {'x': self.x, 'p': self.u, 'ode': self.rhs}
        if integrator == 'collocation':
            opts = {'t0': 0, 'tf': dt, 'number_of_finite_elements': 3, 
                    'simplify': True, 'collocation_scheme': 'radau',
                    'rootfinder':'fast_newton','expand': True,  # fast_newton, newton
                    'interpolation_order': 3}
        elif integrator == 'cvodes':
            opts = {'t0': 0, 'tf': dt, 
                    'nonlinear_solver_iteration': 'functional', # 'expand': True,
                    'linear_multistep_method': 'bdf'}
        I = cs.integrator('I', integrator, dae, opts)
        x_next = I(x0=self.x, p=self.u)["xf"]
        self.F = cs.Function('F', [x, u], [x_next])
        self.dF_dx = cs.Function('dF_dx', [x, u], [cs.jacobian(x_next, x)])
     
    
    def get_acados_model(self)  -> AcadosModel:
        model = AcadosModel()
        model.x = self.x
        model.u = self.u
        model.z = self.z
        model.xdot = self.xdot
            
        model.name = 'iiwa14_MPCmodel'
        model.f_impl_expr = cs.vertcat(self.xdot - self.rhs, self.z - self.rhs_impl)
        model.f_expl_expr = self.rhs
            
        constraint_expr = self.rhs_impl
            
        return model, constraint_expr
        
    def output(self,x):
        return np.array(self.outy(x))
        
    def gravity_torque(self,q):
        dq_ddq = np.zeros(self.nq)
        return np.array(self.rnea(q,dq_ddq,dq_ddq))
    # output (7,1), input (7,1) or (7,) both are ok
    
    def forward_kinemetics(self,q):
        return np.array(self.pee(q)) 
    # the output (3,1), input (7,1) or (7,) both are ok
    
    

if __name__ == "__main__":
    
    # reference joint position for single point
    # q_ref = np.array([-0.44,0.14,2.02,-1.61,0.57,-0.16,-1.37])
    model = Symbolic_model()
    # pee = model.forward_kinemetics(q_ref)
    # print("pee is: ", pee)
    # gravity = model.gravity_torque(q_ref)
    # print("gravity is: ",gravity)
    # center of the circle
    # q_center = np.array([0,-1.58,0,-1.58,0,0,0])
    # p_center = model.forward_kinemetics(q_center)
    # print("p_center is: ", p_center)
    
    
    
    # calculate the circle points to compare the result of inverse kinematcis
    q_circle_ref = np.loadtxt("data/200circle_joint_xy0.1.txt",delimiter=',')
    print("shape:", q_circle_ref.shape)
    pee_circle = np.zeros((q_circle_ref.shape[0],3))
    for i in range(q_circle_ref.shape[0]):
        p_circle = model.forward_kinemetics(q_circle_ref[i,:])
        pee_circle[i,:] = p_circle.reshape(3,)
    
    """
    # show the change of each joint   
    fig,axes = plt.subplots(7,1)
    t = range(q_circle_ref.shape[0])
    axes[0].plot(t, q_circle_ref[:,0],label='q1' 'r')
    axes[1].plot(t, q_circle_ref[:,1],label='q2' 'r')
    axes[2].plot(t, q_circle_ref[:,2],label='q3' 'r')
    axes[3].plot(t, q_circle_ref[:,3],label='q4' 'r')
    axes[4].plot(t, q_circle_ref[:,4],label='q5' 'r')
    axes[5].plot(t, q_circle_ref[:,5],label='q6' 'r')
    axes[6].plot(t, q_circle_ref[:,6],label='q7' 'r')
    plt.show()

    # plot the circle points deduced from the result of the inverse kinematics    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(np.shape(pee_circle)[0]):
        point = pee_circle[i,:]
        ax.scatter(point[0],point[1],point[2], c='b', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    z_gap = np.max(pee_circle[:,2]) - np.min(pee_circle[:,2])
    x_gap = np.max(pee_circle[:,0]) - np.min(pee_circle[:,0])
    y_gap = np.max(pee_circle[:,1]) - np.min(pee_circle[:,1])
    print("z_gap is: ", z_gap)
    print("x_gap is: ", x_gap)
    print("y_gap is: ", y_gap)
    """
  