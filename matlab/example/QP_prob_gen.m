H = [1 -1;-1 2];
c = [-2;-6];
A_eq = [];
b_eq = [];
A_ineq = -[1 1;-1 2;2 1];
b_ineq = -[2;2;3];

casadi_qpProbGen(H,c,A_eq,b_eq,A_ineq,b_ineq,'QP','_C')