function varargout = value_space_QP(H,c,A_eq,b_eq)
% 值空间方法（拉格朗日乘子法）求解有等式约束的二次规划问题:
% min:
%       0.5*x'*H*x + c'*x
% s.t.
%       A_eq*x = b_eq
% 输出:
% x: 近似最优点
% lambda：最优点处的乘子向量
% iter：迭代次数
%
% 需要A是行满秩的

A = [H,-A_eq';-A_eq,zeros(length(b_eq),length(b_eq))];
b = [-c;-b_eq];

[opt,~] = linsolve(A,b);
if norm(A*opt - b,2)/norm(opt,2) > 1e-8 || norm(A*opt - b,2) > 1e-8
    opt = lsqminnorm(A,b);
elseif any(isnan(opt))
    opt = lsqminnorm(A,b);
end

if nargout >= 1
    x = opt(1:length(c));
    varargout{1} = x;
end
if nargout >= 2
    lambda = opt(length(c) + 1:end);
    varargout{2} = lambda;
end
if nargout >= 3
    val = 0.5*x'*H*x + c'*x;
    varargout{3} = val;
end

end