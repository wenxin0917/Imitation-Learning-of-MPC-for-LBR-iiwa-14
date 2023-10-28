%% Recursive Newton-Euler Algorithm for dynamic model of iiwa14 R820
function tau_list = RNE(q,qd,qdd,G)
%% 输入参数：
% q：广义关节坐标，此处为关节转角，1×7矩阵，每一行向量对应一组关节转角，单位：rad；
% qd： 广义关节坐标一阶导数，此处为关节角速度，1×7矩阵，每一行向量对应一组关节角速度，单位：rad/s；
% qdd： 广义关节坐标二阶导数，此处为关节角加速度，1×7矩阵，每一行向量对应一组关节角加速度，单位：rad/s^2；
% G：重力项，当G = 1,有重力影响，当G = 0,无重力影响；
% note:：三个输入参数的长度需保持一致。

%% 输出参数：
% tau_list ：关节力矩，1×7矩阵，每一行向量对应一组关节力矩，单位：Nm

% 判断输入是否符合规则
rows = size(q,1);
if rows ~= size(qd,1) || rows ~= size(qdd,1)
    error("输入参数长度不一致");
end

% 参数初始化
% DH_list：机器人DH参数，4×7矩阵
%          alpha      a     d     theta
dh_list = [0          0    0.1575   q(1);
           -pi/2      0    0        pi+q(2);
           -pi/2      0    0.2045   pi+q(3);
           pi/2       0    0        q(4);
           -pi/2      0    0.1845   pi+q(5);
           pi/2       0    0        q(6);
           -pi/2      0    0.081    pi+q(7);
           0          0    0        0];
       
 % Define symbolic mass_list as a vector
mass_list = sym('mass_list', [1, 7]);

% Define symbolic mass_center_list as a matrix
mass_center_list = sym('mass_center_list', [7, 3]);

% Define symbolic inertia_tensor_list as a 3x3x7 symbolic array
inertia_tensor_list = sym('inertia_tensor_list', [3, 3, 7]);
       
% mass_list: 连杆的质量，1×7矩阵，单位：kg       
mass_list = [4 4 3 2.7 1.7 1.8 0.3];

% mass_center_list：连杆质心在连杆坐标系下的位置，3×7矩阵，单位：m                             
%                   x         y           z
mass_center_list = [0        -0.03       0.12;
                    0.0003   0.059       0.042;
                    0        0.03        0.13;
                    0        0.067       0.034;
                    0.0001   0.021       0.076;
                    0        0.0006      0.0004;
                    0        0           0.02];
                
% inertia_tensor_list：连杆关于质心坐标系的惯性张量，质心坐标系与连杆坐标系方位一致，7个3×3矩阵，单位kg*m^2
%         I                =       Ixx            -Ixy           -Ixz
%                                 -Ixy             Iyy           -Iyz
%                                 -Ixz            -Iyz            Izz
inertia_tensor_list(:,:,1) = [0.1       0        0;
                              0         0.09     0;
                              0         0      0.02];
                          
inertia_tensor_list(:,:,2) = [0.05       0         0;
                              0          0.018     0;
                              0          0      0.044];
                          
inertia_tensor_list(:,:,3) = [0.08       0        0;
                              0          0.075    0;
                              0          0      0.01];
                          
inertia_tensor_list(:,:,4) = [0.03      0         0;
                              0         0.01      0;
                              0         0      0.029]; 
                          
inertia_tensor_list(:,:,5) = [0.02       0          0;
                              0          0.018      0;
                              0          0      0.005];
                          
inertia_tensor_list(:,:,6) = [0.005       0           0;
                              0           0.0036      0;
                              0           0     0.0047];
                          
inertia_tensor_list(:,:,7) = [0.001         0            0;
                              0             0.001       0;
                              0             0           0.001];
                          
% f_external：施加在末端连杆的外力和外力矩
f_external = sym('f_external', [2, 3]);
% f_external = zeros(2, 3);

number_of_links = 7;
z = sym('z',[3,1]);
z = [0,0,1]';  % 关节轴
%%判断是否施加重力
if G == 1
    g = 9.81;     % 重力加速度，单位m/s^2
else
    g = 0;
end

% 位姿变换矩阵参数设置
for i = 1:number_of_links+1
    dh = dh_list(i,:);
    alpha(i) = dh(1);
    a(i) = dh(2);
    d(i) = dh(3);
    theta(i) = dh(4);
    T(:,:,i) = [cos(theta(i)),            -sin(theta(i)),           0,           a(i);
            sin(theta(i))*cos(alpha(i)), cos(theta(i))*cos(alpha(i)), -sin(alpha(i)), -sin(alpha(i))*d(i);
            sin(theta(i))*sin(alpha(i)), cos(theta(i))*sin(alpha(i)), cos(alpha(i)), cos(alpha(i))*d(i);
            0,                     0,                     0,          1];
    T = T(:,:,i);
    % 提取旋转矩阵并求逆
    R(:,:,i) = T(1:3,1:3);
    R_inv(:,:,i) = inv(T(1:3,1:3));
    P(:,i) = T(1:3,4:4);
end

% 外推 --->
for i = 0:number_of_links-1
    if i == 0
       wi = [0,0,0]';      % 初始角速度为0
       dwi = [0,0,0]';     % 初始角加速度为0
       dvi = [0, 0, g]';   % 初始加速度，根据坐标系0对重力加速度方向进行设置
    else
        wi = w(:,i);
        dwi = dw(:,i);
        dvi = dv(:,i);
    end
    w(:,i+1) = R(:,:,i+1)*wi + qd(i+1)*z;
    dw(:,i+1) = R(:,:,i+1)*dwi + cross(R(:,:,i+1)*wi,qd(i+1)*z) + qdd(i+1)*z;
    dv(:,i+1) = R(:,:,i+1)*(cross(dwi,P(:,i+1)) + cross(wi,cross(wi,P(:,i+1))) + dvi);
    dvc(:,i+1) = cross(dw(:,i+1),mass_center_list(i+1,:)')...
                        + cross(w(:,i+1),cross(w(:,i+1),mass_center_list(i+1,:)'))...
                        + dv(:,i+1);
     F(:,i+1) = mass_list(i+1)*dvc(:,i+1);
     N(:,i+1) = inertia_tensor_list(:,:,i+1)*dw(:,i+1) + cross(w(:,i+1),inertia_tensor_list(:,:,i+1)*w(:,i+1));
end
% 内推 <---
for i = number_of_links:-1:1
    if i == number_of_links
       f(:,i+1) = f_external(1,:)';
       n(:,i+1) = f_external(2,:)';
    end
    f(:,i) = R_inv(:,:,i+1)*f(:,i+1) + F(:,i);
    n(:,i) = N(:,i) + R_inv(:,:,i+1)*n(:,i+1) + cross(mass_center_list(i,:)',F(:,i))...
                    + cross(P(:,i+1),R_inv(:,:,i+1)*f(:,i+1));
    tau_list (i) = dot(n(:,i),z);
end
