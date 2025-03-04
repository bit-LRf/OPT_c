function casadi_nlpProbGen(x,p,f,eq,ineq,funName,suffix)
% this is a function for generating casadi C code function
%   x: variables
%   p: parameters
%   f: optimize object
%   eq: equation constrains
%   ineq: inequation constrains
%   funName: user-defined function name ('string' data type)
%   suffix: the file name suffix (not the real suffix) after 'funName' (also 'string' data type)
% for example, if funName = 'hello', sufix = '_world', 
% then the generated code is hello_world.h and hello_world.cpp

if nargin < 6
    funName = 'fun';
end
if nargin < 7
    suffix = '_C';
end

n_x = length(x);
n_eq = length(eq);
n_ineq = length(ineq);

if n_x == 0
    error('no variables, n_x = 0\n');
end

if n_ineq ~= 0
    s = casadi.SX.sym('s',n_ineq,1);

    x = [x;s];
    n_x = length(x);

    eq = [eq;ineq - s];
    n_eq = length(eq);
end

fun_f = casadi.Function([funName,'_f'],{x,p},{f});

df = jacobian(f,x);
df = df + zeros(1,n_x);
fun_df = casadi.Function([funName,'_df'],{x,p},{df});

fun_eq = casadi.Function([funName,'_eq'],{x,p},{eq});

deq = jacobian(eq,x);
fun_deq = casadi.Function([funName,'_deq'],{x,p},{deq});

lambda = casadi.SX.sym('lambda',n_eq,1);
lag_f = f;
if n_eq ~= 0
    lag_f = lag_f - lambda'*eq;
end
lag_h = hessian(lag_f,x);
fun_h_lower = casadi.Function([funName,'_h_lower'],{x,lambda,p},{tril(lag_h)});

c_opts = struct();
c_opts.cpp = true;
c_opts.with_header = true;
c_opts.casadi_int = 'int';

srv = casadi.CodeGenerator([funName,suffix],c_opts);
srv.add(fun_f);
srv.add(fun_df);
srv.add(fun_eq);
srv.add(fun_deq);
srv.add(fun_h_lower);
srv.generate();

end