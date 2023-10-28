
%% deduce the forward kinematics equation
syms q1 q2 q3 q4 q5 q6 q7 real
syms dq1 dq2 dq3 dq4 dq5 dq6 dq7 real
syms ddq1 ddq2 ddq3 ddq4 ddq5 ddq6 ddq7 real


q = [q1;q2;q3;q4;q5;q6;q7];
qd = [dq1;dq2;dq3;dq4;dq5;dq6;dq7];
qdd = [ddq1;ddq2;ddq3;ddq4;ddq5;ddq6;ddq7];


G = 1;

tau_list = RNE(q,qd,qdd,G);

% Define symbolic inertia matrix M, Coriolis matrix C, and gravitational vector G as separate variables
M = sym('M', [7, 7]);
C = sym('C', [7, 7]);
G = sym('G', [7, 1]);

% Here, M represents the inertia matrix, C represents the Coriolis and centrifugal terms,
% and G represents the gravitational terms.

% Define the symbolic equation tau = M * ddq + C * dq + G
eq = tau_list' - M * [ddq1; ddq2; ddq3; ddq4; ddq5; ddq6; ddq7] - C * [dq1; dq2; dq3; dq4; dq5; dq6; dq7] - G;

% Solve the equation for M, C, and G element by element
M_elements = sym('M', [7, 7]);
C_elements = sym('C', [7, 7]);
G_elements = sym('G', [7, 1]);

for i = 1:7
    for j = 1:7
        M_elements(i, j) = solve(eq(i), M_elements(i, j));
        C_elements(i, j) = solve(eq(i), C_elements(i, j));
    end
    G_elements(i) = solve(eq(i), G_elements(i));
end

% Now you have M, C, and G as symbolic expressions element by element.

% Display the symbolic results
disp('Inertia Matrix M (element by element):');
disp(M_elements);

disp('Coriolis and Centrifugal Matrix C (element by element):');
disp(C_elements);

disp('Gravitational Vector G (element by element):');
disp(G_elements);