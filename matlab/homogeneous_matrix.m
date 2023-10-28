%{
attention : the world coordinate is the same as the joint 0 
%}

%% calculate each homogeneous matrix

% Define symbolic DH parameters
syms q1 q2 q3 q4 q5 q6 q7 real
syms dq1 dq2 dq3 dq4 dq5 dq6 dq7
syms ddq1 ddq2 ddq3 ddq4 ddq5 ddq6 ddq7
syms alpha1 beta1 gama1

% Create a symbolic DH parameter table
DH_table = [
    0.1575, q1, 0, 0;
    0, pi + q2, 0, -pi/2;
    0.2045, pi + q3, 0, -pi/2;
    0, q4, 0, pi/2;
    0.1845, pi + q5, 0, -pi/2;
    0, q6, 0, pi/2;
    0.081, pi + q7, 0, -pi/2;
];

% Initialize an array to store transformation matrices
T_matrices = cell(1, 7);

% Compute the transformation matrices and store them
T = eye(4);
for i = 1:7
    d = DH_table(i, 1);
    theta = DH_table(i, 2);
    a = DH_table(i, 3);
    alpha = DH_table(i, 4);
    
    A = [
        cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta);
        sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta);
        0, sin(alpha), cos(alpha), d;
        0, 0, 0, 1
    ];
    
    T = T * A;
    
    % Store the transformation matrix in the array
    T_matrices{i} = A;
end

disp(T);

% T_matrices is a cell array containing transformation matrices for different joint configurations
% Access the transformation matrices like T_matrices{1}, T_matrices{2}, etc.

% To display and save individual transformation matrices, you can use a loop
for i = 1:7
    disp(['Transformation Matrix T', num2str(i), ':']);
    disp(T_matrices{i});
    
    % If you want to save each T to a separate file (e.g., T1.mat, T2.mat, etc.)
    % save(['T', num2str(i), '.mat'], 'T_matrices{i}');
end

%% calculate jacobian matrix
% get f_q to represent the end-effect position in the world 
x_ee = [0; 0; 0.045];
x_rotation = [alpha1;beta1;gama1];
x_world_4 = T *[x_ee;1];
x_world = x_world(1:3,:);
x_rotation_world = T(1:3,1:3)* x_rotation;
f_q = [x_world;x_rotation_world]

% Print the shape of f_q
fprintf('Shape of f_q: %dx%d\n', size(f_q, 1), size(f_q, 2));

% Calculate the Jacobian matrix using the chain rule
J = jacobian(f_q, [q1, q2, q3, q4, q5, q6, q7]);
% Print the shape of f_q
fprintf('Shape of Jacobian: %dx%d\n', size(J, 1), size(J, 2));
% Display the Jacobian matrix
disp('Jacobian Matrix:');
disp(J);


