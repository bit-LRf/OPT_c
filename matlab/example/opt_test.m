% 线性规划
% min:
%       c'*x
% s.t.
%       A_eq*x = b_eq
%       A_ineq*x >= b_ineq
%       x >= 0

% c = [-10;-15];
% A_eq = [];
% b_eq = [];
% A_ineq = [-2 -3;-1 -2];
% b_ineq = [-100;-50];
% lbx = [0;0];
% ubx = [inf;inf];

c = -[5000;4000];
A_eq = [];
b_eq = [];
A_ineq = -[0.5 0.2;0.4 0.3];
b_ineq = -[200;150];
lbx = [0;0];
ubx = [100;300];

sol = lp_interiorPoint(c,A_eq,b_eq,A_ineq,b_ineq,lbx,ubx)
sol.x
