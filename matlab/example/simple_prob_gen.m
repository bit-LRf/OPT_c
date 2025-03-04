% note: this problem is infeasible when: ubx < 2 since sum(x) = 10, if you
% set ubx = 2, the solver also performs bad.

x = casadi.SX.sym('x',5);
p = casadi.SX.sym('p',2);

f = sum(x.^2) + p(1)*x(2)*x(4) + p(2)*x(1)*x(3);

eq = [sum(x) - 10;x(1) - x(2);x(1) - x(2)];

casadi_nlpProbGen(x,p,f,eq,[],'testProb','_yourSuffix');