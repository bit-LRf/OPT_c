function sol = lp_interiorPoint(c,A_eq,b_eq,A_ineq,b_ineq,lbx,ubx)
% lp_interiorPoint(c,A_eq,b_eq,A_ineq,b_ineq,param)
% 内点法求解线性规划问题:
% min:
%       c'*x
% s.t.
%       A_eq*x = b_eq
%       A_ineq*x >= b_ineq
%       x >= 0
% 
% 输出:
% sol.x
% sol.lambda_eq
% sol.lambda_ineq
% sol.lambda_lbx
% sol.lambda_ubx
% sol.val
% sol.iter

% 参数
iter_max = 100;
beta = 0.1;
accept_tol = 1e-6;
kappa_1 = 0.01;
kappa_2 = 0.01;

% 修改为标准问题
n_x = length(c);
if isempty(lbx)
    lbx = -inf(n_x,1);
end
if isempty(ubx)
    ubx = inf(n_x,1);
end
n_eq = length(b_eq);
n_ineq = length(b_ineq);
if n_ineq ~= 0
    A_eq = [A_eq,zeros(n_eq,n_ineq);
        A_ineq,-eye(n_ineq)];
    b_eq = [b_eq;b_ineq];
    c = [c;zeros(n_ineq,1)];
    lbx = [lbx;zeros(n_ineq,1)];
    ubx = [ubx;inf(n_ineq,1)];
end

% 检查有效约束并生成索引
idx_lbx = ~isinf(lbx);
idx_ubx = ~isinf(ubx);

% 测量
n_x = length(c);
n_eq = length(b_eq);
n_lbx = sum(idx_lbx);
n_ubx = sum(idx_ubx);

% 计算初始值
[x_0,lambda_0] = value_space_QP(eye(n_x),c,A_eq,b_eq);
s_lbx_0 = zeros(n_x,1);
s_ubx_0 = zeros(n_x,1);
s_lbx_0(idx_lbx) = 1;
s_ubx_0(idx_ubx) = 1;

% 修正可行性
% 仅下界
idx_lbx_only = and(idx_lbx,~idx_ubx);
lbx_only = lbx(idx_lbx_only);
x_0(idx_lbx_only) = max(x_0(idx_lbx_only),lbx_only + kappa_1*max(1,abs(lbx_only)));

% 仅上界
idx_ubx_only = and(idx_ubx,~idx_lbx);
ubx_only = ubx(idx_ubx_only);
x_0(idx_ubx_only) = min(x_0(idx_ubx_only),ubx_only - kappa_1*max(1,abs(ubx_only)));

% 上下界
idx_both = and(idx_lbx,idx_ubx);
lbx_both = lbx(idx_both);
ubx_both = ubx(idx_both);
tmp_lb = min(kappa_1*max(1,abs(lbx_both)),kappa_2*(ubx_both - lbx_both));
tmp_ub = min(kappa_1*max(1,abs(ubx_both)),kappa_2*(ubx_both - lbx_both));
x_0(idx_both) = min(max(x_0(idx_both),lbx_both + tmp_lb),ubx_both - tmp_ub);

% 开始计算
x = x_0;
lambda = lambda_0;
s_lbx = s_lbx_0;
s_ubx = s_ubx_0;
iter = 0;
while 1
    % 测量KKT系统
    lbx_active = x(idx_lbx) - lbx(idx_lbx);
    ubx_active = -x(idx_ubx) + ubx(idx_ubx);
    
    r_c = A_eq'*lambda + s_lbx - s_ubx - c;
    r_b = A_eq*x - b_eq;
    r_lbxs = lbx_active.*s_lbx(idx_lbx);
    r_ubxs = ubx_active.*s_ubx(idx_ubx);

    % 停机条件
    if norm([r_b;r_c;r_lbxs;r_ubxs],inf) <= accept_tol
        break;
    end

    % 预测步
    % 对偶测量
    mu = (sum(r_lbxs) + sum(r_ubxs))/(n_lbx + n_ubx);

    % 求解KKT系统
    D = zeros(n_x,1);
    D(idx_lbx) = s_lbx(idx_lbx)./lbx_active;
    D(idx_ubx) = D(idx_ubx) + s_ubx(idx_ubx)./ubx_active;
    reduced_KKT_A = [-diag(D),A_eq';A_eq,zeros(n_eq)];
    dA = decomposition(reduced_KKT_A,'lu');

    R_lbx = zeros(n_x,1);
    R_ubx = zeros(n_x,1);
    R_lbx(idx_lbx) = r_lbxs./lbx_active;
    R_ubx(idx_ubx) = r_ubxs./ubx_active;
    reduced_KKT_B = [-r_c + R_lbx - R_ubx;-r_b];

    tmp_sol = dA\reduced_KKT_B;
    diff_x_aff = tmp_sol(1:n_x);
    diff_s_lbx_aff = zeros(n_x,1);
    diff_s_lbx_aff(idx_lbx) = -s_lbx(idx_lbx) - s_lbx(idx_lbx).*diff_x_aff(idx_lbx)./lbx_active;
    diff_s_ubx_aff = zeros(n_x,1);
    diff_s_ubx_aff(idx_ubx) = -s_ubx(idx_ubx) + s_ubx(idx_ubx).*diff_x_aff(idx_ubx)./ubx_active;

    % 求解步长
    tmp_idx_primal_lbx = and(diff_x_aff < 0,idx_lbx);
    tmp_idx_primal_ubx = and(diff_x_aff > 0,idx_ubx);
    alpha_primal_aff = 1;
    if any(tmp_idx_primal_lbx)
        alpha_primal_aff = min(alpha_primal_aff,min(-(x(tmp_idx_primal_lbx) - lbx(tmp_idx_primal_lbx))./diff_x_aff(tmp_idx_primal_lbx)));
    end
    if any(tmp_idx_primal_ubx)
        alpha_primal_aff = min(alpha_primal_aff,min((-x(tmp_idx_primal_ubx) + ubx(tmp_idx_primal_ubx))./diff_x_aff(tmp_idx_primal_ubx)));
    end

    tmp_idx_dual_lbx = and(diff_s_lbx_aff < 0,idx_lbx);
    tmp_idx_dual_ubx = and(diff_s_ubx_aff < 0,idx_ubx);
    alpha_dual_aff = 1;
    if any(tmp_idx_dual_lbx)
        alpha_dual_aff = min(alpha_dual_aff,min(-s_lbx(tmp_idx_dual_lbx)./diff_s_lbx_aff(tmp_idx_dual_lbx)));
    end
    if any(tmp_idx_dual_ubx)
        alpha_dual_aff = min(alpha_dual_aff,min(-s_ubx(tmp_idx_dual_ubx)./diff_s_ubx_aff(tmp_idx_dual_ubx)));
    end

    lbx_aff = lbx_active + alpha_primal_aff*diff_x_aff(idx_lbx);
    ubx_aff = ubx_active - alpha_primal_aff*diff_x_aff(idx_ubx);
    s_lbx_aff = s_lbx(idx_lbx) + alpha_dual_aff*diff_s_lbx_aff(idx_lbx);
    s_ubx_aff = s_ubx(idx_ubx) + alpha_dual_aff*diff_s_ubx_aff(idx_ubx);

    % 校正步
    % 对偶测量
    mu_aff = [lbx_aff;ubx_aff]'*[s_lbx_aff;s_ubx_aff]/(n_lbx + n_ubx);

    % 求解KKT系统
    sigma = (mu_aff/mu)^3;
    r_lbxs = r_lbxs + diff_x_aff(idx_lbx).*diff_s_lbx_aff(idx_lbx) - sigma*mu;
    r_ubxs = r_ubxs - diff_x_aff(idx_ubx).*diff_s_ubx_aff(idx_ubx) - sigma*mu;
    R_lbx = zeros(n_x,1);
    R_ubx = zeros(n_x,1);
    R_lbx(idx_lbx) = r_lbxs./lbx_active;
    R_ubx(idx_ubx) = r_ubxs./ubx_active;
    reduced_KKT_B = [-r_c + R_lbx - R_ubx;-r_b];

    tmp_sol = dA\reduced_KKT_B;
    diff_x = tmp_sol(1:n_x);
    diff_lambda = tmp_sol(n_x + 1:end);
    diff_s_lbx = zeros(n_x,1);
    diff_s_lbx(idx_lbx) = -R_lbx(idx_lbx) - s_lbx(idx_lbx).*diff_x(idx_lbx)./lbx_active;
    diff_s_ubx = zeros(n_x,1);
    diff_s_ubx(idx_ubx) = -R_ubx(idx_ubx) + s_ubx(idx_ubx).*diff_x(idx_ubx)./ubx_active;

    % 求解步长
    eta = 1 - beta^(iter + 1);
    
    tmp_idx_primal_lbx = and(diff_x < 0,idx_lbx);
    tmp_idx_primal_ubx = and(diff_x > 0,idx_ubx);
    alpha_primal = 1;
    if any(tmp_idx_primal_lbx)
        alpha_primal = min(alpha_primal,eta*min(-(x(tmp_idx_primal_lbx) - lbx(tmp_idx_primal_lbx))./diff_x(tmp_idx_primal_lbx)));
    end
    if any(tmp_idx_primal_ubx)
        alpha_primal = min(alpha_primal,eta*min((-x(tmp_idx_primal_ubx) + ubx(tmp_idx_primal_ubx))./diff_x(tmp_idx_primal_ubx)));
    end

    tmp_idx_dual_lbx = and(diff_s_lbx < 0,idx_lbx);
    tmp_idx_dual_ubx = and(diff_s_ubx < 0,idx_ubx);
    alpha_dual = 1;
    if any(tmp_idx_dual_lbx)
        alpha_dual = min(alpha_dual,eta*min(-s_lbx(tmp_idx_dual_lbx)./diff_s_lbx(tmp_idx_dual_lbx)));
    end
    if any(tmp_idx_dual_ubx)
        alpha_dual = min(alpha_dual,eta*min(-s_ubx(tmp_idx_dual_ubx)./diff_s_ubx(tmp_idx_dual_ubx)));
    end

    % 迭代
    x = x + alpha_primal*diff_x;
    lambda = lambda + alpha_primal*diff_lambda;
    s_lbx = s_lbx + alpha_dual*diff_s_lbx;
    s_ubx = s_ubx + alpha_dual*diff_s_ubx;

    % 检测迭代次数
    iter = iter + 1;
    if iter >= iter_max
        break
    end
end

% 还原
c = c(1:end - n_ineq);
x = x(1:end - n_ineq);
lambda_lbx = s_lbx(1:end - n_ineq);
lambda_ubx = s_ubx(1:end - n_ineq);
lambda_eq = lambda(1:end - n_ineq);
lambda_ineq = s_lbx(end - n_ineq + 1:end);

% 输出
sol.x = x;
sol.lambda_eq = lambda_eq;
sol.lambda_ineq = lambda_ineq;
sol.lambda_lbx = lambda_lbx;
sol.lambda_ubx = lambda_ubx;
sol.val = c'*x;
sol.iter = iter;

end