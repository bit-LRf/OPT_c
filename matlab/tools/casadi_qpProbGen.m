function casadi_qpProbGen(H,c,A_eq,b_eq,A_ineq,b_ineq,funName,suffix)
% min x'*H*x + c'*x
% s.t. A_eq*x = b_eq
%       A_ineq*x >= b_ineq
%       lbx <= x <= ubx

if nargin < 6
    funName = 'fun';
end
if nargin < 7
    suffix = '_C';
end

H = sparse(H);
A_eq = sparse(A_eq);
A_ineq = sparse(A_ineq);

n_x = size(H,1);
n_eq = size(A_eq,1);
n_ineq = size(A_ineq,1);

x = casadi.SX.sym('x',n_x,1);
p = [];

f = 1/2*x'*H*x + c'*x;
if n_eq == 0
    eq = [];
else
    eq = A_eq*x - b_eq;
end
if n_ineq == 0
    ineq = [];
else
    ineq = A_ineq*x - b_ineq;
end

casadi_nlpProbGen(x,p,f,eq,ineq,funName,suffix);

end