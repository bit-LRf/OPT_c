% road parameters
step_s = 0.1;

% road = Generate_LinCurv_turn2([0;40;0],step_s);
% road = Generate_LinCurv_turn2([0 1 1 0;20 30 40 30;0 60 50 0],step_s);
% road = Generate_LinCurv_turn2([0 1 1 1 -1 0;20 2 12 2 20 10;0 8 8 8 8 0],step_s);
% road = Generate_LinCurv_turn2([0 1 0;15 50 15;0 8 0],step_s);
road = Generate_LinCurv_turn2([0 1 0;15 25 15;0 4 0],step_s);

curv_interp = griddedInterpolant(road.S,road.curv,'linear','nearest');
phi_interp = griddedInterpolant(road.S,road.phi,'linear','nearest');

% state variabels
X = casadi.SX.sym('state',7,1);
n_x = length(X);

% control variables
U = casadi.SX.sym('control',5,1);
n_u = length(U);

% parameters
p = casadi.SX.sym('param',2,1);
n_p = length(p);

% constraint
ubx = [inf;0.52;inf;2;0.78;inf;inf];
lbx = [2;-0.52;-inf;-2;-0.78;-inf;-inf];

ubu = [0.52;0.8;0.8;0.8;0.8];
lbu = -[0.52;0.8;0.8;0.8;0.8];

% initial state
X_0 = [10;0;0;0;0;0;0];
U_0 = [0;0;0;0;0];

% warm start value
X_ws = X_0;
U_ws = [0;0;0;0;0];

% lap info
N = 40;
ds = road.S(end)/N;
S = cumtrapz(ds*ones(1,N));
curv = curv_interp(S);

% loop over the lap
r = {};
r_ptr = 1;

eq = {};
eq_ptr = 1;

ineq = {};
ineq_ptr = 1;

f = 0;
X_k = X_0;
U_k = U_0;
for k = 1:N
    % 
    X_kn1 = X_k;
    U_kn1 = U_k;

    % 创建新的变量
    U_k = casadi.SX.sym(['U_',num2str(k - 1)],n_u,1);
    r{r_ptr} = U_k;
    r_ptr = r_ptr + 1;

    X_k = casadi.SX.sym(['X_',num2str(k)],n_x,1);
    r{r_ptr} = X_k;
    r_ptr = r_ptr + 1;

    % 差分近似的动力学约束
    [dx,df] = fc2_model(X_k,U_k,curv(k));
    
    f = f + df*ds;

    % 正则化项
    f = f + p(1)*sum(((U_k - U_kn1)/ds).^2)*ds;
    f = f + p(2)*sum(((X_k - X_kn1)/ds).^2)*ds;

    %
    eq{eq_ptr} = dx*ds - (X_k - X_kn1);
    eq_ptr = eq_ptr + 1;
end

% 建立nlp问题
r = vertcat(r{:});
eq = vertcat(eq{:});
ineq = vertcat(ineq{:});
casadi_nlpProbGen(r,p,f,eq,ineq,'largeProb','_C');


%% 
function Track = Generate_LinCurv_turn2(properties,step_s)
% --------------INPUTS---------------
% properties 
% 长度：3×n
% 列属性从上到下分别是：弯道方向（0：直线，1：左转，-1：右转）、长度、最小半径
% 
% step_s
% 路径的离散距离

% 曲率曲线生成
curv_max = properties(1,:)./properties(3,:);

n = length(properties(1,:));

curv = zeros(1,sum(properties(2,:))/step_s);

dis_tmp = 0;

for i = 1:n
    if properties(1,i) == 0
        dis_tmp = dis_tmp + properties(2,i);
    elseif properties(1,i)^2 == 1
        pointer = [dis_tmp/step_s + 1,(dis_tmp + properties(2,i))/step_s];

        if i > 1 && properties(1,i)*properties(1,i - 1) > 0
            curv_init = curv_max(i - 1);

            pointer(1) = pointer(1) - 1;
        else
            curv_init = 0;
        end

        if i < n && properties(1,i)*properties(1,i + 1) > 0
            curv_middle = (curv_max(i) + curv_init)/2;

            curv_end = curv_max(i);
        else
            curv_middle = curv_max(i);

            if curv_middle == curv_init
                curv_middle = 0.5*curv_middle;
            end

            curv_end = 0;
        end

        curv(pointer(1):floor((pointer(2) + pointer(1))/2)) = curv_init ...
            + 2*(curv_middle - curv_init)/(pointer(2) - pointer(1))*((pointer(1):floor((pointer(2) + pointer(1))/2)) - pointer(1));

        curv(ceil((pointer(2) + pointer(1))/2):pointer(2)) = curv_middle ...
            + 2*(curv_end - curv_middle)/(pointer(2) - pointer(1))*((ceil((pointer(2) + pointer(1))/2):pointer(2)) - (pointer(2) + pointer(1))/2);

        dis_tmp = dis_tmp + properties(2,i);
    else
        fprintf("错误的弯道属性")
        return
    end
end

% 主体
D_s = step_s*ones(size(curv));

% phi是曲率curv在单位路径上的积分
phi = cumtrapz(D_s.*curv);

% 路径的x-y坐标系, 从起点(0,0)开始
Track.x = cumtrapz(D_s.*cos(phi));
Track.y = cumtrapz(D_s.*sin(phi));
Track.phi = phi;

% 路径的s-curv坐标系
Track.curv = curv;
Track.S = cumtrapz(D_s);
Track.N = length(Track.x);

% 左右边界
half_weigth = 2 + 1.58/2;

left_upper = curv < 0;
right_upper = curv > 0;

Track.left_x = Track.x - half_weigth*(1 + left_upper.*abs(curv)*1).*sin(phi);
Track.right_x = Track.x + half_weigth*(1 + right_upper.*abs(curv)*1).*sin(phi);

Track.left_y = Track.y + half_weigth*(1 + left_upper.*abs(curv)*1).*cos(phi);
Track.right_y = Track.y - half_weigth*(1 + right_upper.*abs(curv)*1).*cos(phi);

end


%% 
function [D_X_ds,D_J_ds] = fc2_model(X,U,curv)
i = 1;
v = X(i);i = i + 1;
beta = X(i);i = i + 1;
omega = X(i);i = i + 1;
e = X(i);i = i + 1;
erro_PHI = X(i);i = i + 1;
load_trans_x = X(i);i = i + 1;
load_trans_y = X(i);i = i + 1;

i = 1;
delta = U(i);i = i + 1;
unit_F_x_1 = U(i);i = i + 1;
unit_F_x_2 = U(i);i = i + 1;
unit_F_x_3 = U(i);i = i + 1;
unit_F_x_4 = U(i);i = i + 1;

mass = 1100;
inertial_z = 1343.1;
l_f = 1.04;
l_r = 1.56;
l_x = l_f + l_r;
l_y = 1.89;
h_cg = 0.54;
g = 9.8;
tau = 0.1;

% tire tramac
% B = 6.8488;
% C = 1.4601;
% D = 1;
% E = -3.6121;

% tire tarmac,wet
% B = 11.415;
% C = 1.4601;
% D = 0.6;
% E = -0.20939;

% tire gravel
% B = 15.289;
% C = 1.0901;
% D = 0.6;
% E = 0.86215;

% carsim fit 1
% B = 10.71;
% C = 1.265;
% D = 0.9446;
% E = -1.185;

% MF52 fit 2
B = 13.35;
C = 1.009;
D = 0.9446;
E = 0.0898;

mu_x = D;

%
F_z_1 = mass*(l_r*g/(2*l_x) - h_cg*load_trans_x/(2*l_x) - h_cg*load_trans_y/(2*l_y));
F_z_2 = mass*(l_r*g/(2*l_x) - h_cg*load_trans_x/(2*l_x) + h_cg*load_trans_y/(2*l_y));
F_z_3 = mass*(l_f*g/(2*l_x) + h_cg*load_trans_x/(2*l_x) - h_cg*load_trans_y/(2*l_y));
F_z_4 = mass*(l_f*g/(2*l_x) + h_cg*load_trans_x/(2*l_x) + h_cg*load_trans_y/(2*l_y));

delta = [delta*ones(2,1);zeros(2,1)];
F_z = [F_z_1;F_z_2;F_z_3;F_z_4];
unit_F_x = [unit_F_x_1;unit_F_x_2;unit_F_x_3;unit_F_x_4];

%
s_y = wheel_slip(v,beta,omega,delta,l_f,l_r,l_y);
F_y_pure = MF98(s_y,F_z,D,C,B,E);

F_y = F_y_pure.*sqrt(1 - min(unit_F_x.^2,1 - 1e-3));
% F_y = F_y_pure;

sub_X = [v;beta;omega];
sub_U = [delta;mu_x*unit_F_x.*F_z;F_y];
D_sub_X = vehicle_model(sub_X,sub_U,mass,inertial_z,l_f,l_r,l_y);

D_v_dt = D_sub_X(1);
D_beta_dt = D_sub_X(2);
D_omega_dt = D_sub_X(3);

D_s_dt = v*cos(erro_PHI)/(1 - curv*e);
D_e_dt = v*sin(erro_PHI);
D_erro_PHI_dt = D_beta_dt + omega - curv*D_s_dt;

a_x = D_v_dt*cos(beta) - (D_beta_dt + omega)*v*sin(beta);
a_y = (D_beta_dt + omega)*v*cos(beta) + D_v_dt*sin(beta);

D_load_trans_x = 1/tau*(a_x - load_trans_x);
D_load_trans_y = 1/tau*(a_y - load_trans_y);

%% 
D_X_ds = [D_v_dt;D_beta_dt;D_omega_dt;D_e_dt;D_erro_PHI_dt;D_load_trans_x;D_load_trans_y]./D_s_dt;

D_J_ds = 1/D_s_dt + 0.001*beta^2;

end


%%
function s_y = wheel_slip(v, beta, omega, delta, l_f, l_r, l_y)
v_x_center = casadi.SX.zeros(4,1);
v_x_center(1) = v*cos(beta - delta(1)) + omega*l_f*sin(delta(1)) - omega*l_y/2*cos(delta(1));% 左前
v_x_center(2) = v*cos(beta - delta(2)) + omega*l_f*sin(delta(2)) + omega*l_y/2*cos(delta(2));% 右前
v_x_center(3) = v*cos(beta - delta(3)) - omega*l_r*sin(delta(3)) - omega*l_y/2*cos(delta(3));% 左后
v_x_center(4) = v*cos(beta - delta(4)) - omega*l_r*sin(delta(4)) + omega*l_y/2*cos(delta(4));% 右后

v_y_slip = casadi.SX.zeros(4,1);
v_y_slip(1) = -(v*sin(beta - delta(1)) + omega*l_f*cos(delta(1)) + omega*l_y/2*sin(delta(1)));
v_y_slip(2) = -(v*sin(beta - delta(2)) + omega*l_f*cos(delta(2)) - omega*l_y/2*sin(delta(2)));
v_y_slip(3) = -(v*sin(beta - delta(3)) - omega*l_r*cos(delta(3)) + omega*l_y/2*sin(delta(3)));
v_y_slip(4) = -(v*sin(beta - delta(4)) - omega*l_r*cos(delta(4)) - omega*l_y/2*sin(delta(4)));

s_y = v_y_slip./v_x_center;

end


%% 
function F_y = MF98(s_y,F_z, D, C, B, E)
Mu_y = D*sin(C*atan(s_y*B - E*(s_y*B - atan(s_y*B))));
F_y = Mu_y.*F_z;

end


%% 
function D_X = vehicle_model(X,U, mass, inertial_z, l_f, l_r, l_y)
i = 1;
v = X(i);i = i + 1;
beta = X(i);i = i + 1;
omega = X(i);i = i + 1;

i = 1;
delta_f_1 = U(i);i = i + 1;
delta_f_2 = U(i);i = i + 1;
delta_r_1 = U(i);i = i + 1;
delta_r_2 = U(i);i = i + 1;

F_x_f_1 = U(i);i = i + 1;
F_x_f_2 = U(i);i = i + 1;
F_x_r_1 = U(i);i = i + 1;
F_x_r_2 = U(i);i = i + 1;

F_y_f_1 = U(i);i = i + 1;
F_y_f_2 = U(i);i = i + 1;
F_y_r_1 = U(i);i = i + 1;
F_y_r_2 = U(i);i = i + 1;

% 车体动力学模型
D_v = 1/mass*(...
    F_x_f_1*cos(delta_f_1 - beta) + F_x_f_2*cos(delta_f_2 - beta) ...
    + F_x_r_1*cos(delta_r_1 - beta) + F_x_r_2*cos(delta_r_2 - beta) ...
    - F_y_f_1*sin(delta_f_1 - beta) - F_y_f_2*sin(delta_f_2 - beta) ...
    - F_y_r_1*sin(delta_r_1 - beta) - F_y_r_2*sin(delta_r_2 - beta));

D_beta = 1/(mass*v)*(...
    F_x_f_1*sin(delta_f_1 - beta) + F_x_f_2*sin(delta_f_2 - beta) ...
    + F_x_r_1*sin(delta_r_1 - beta) + F_x_r_2*sin(delta_r_2 - beta) ...
    + F_y_f_1*cos(delta_f_1 - beta) + F_y_f_2*cos(delta_f_2 - beta) ...
    + F_y_r_1*cos(delta_r_1 - beta) + F_y_r_2*cos(delta_r_2 - beta)) - omega;

D_omega = 1/inertial_z*(...
    l_y/2*(-F_x_f_1*cos(delta_f_1) + F_y_f_1*sin(delta_f_1)) ...
    + l_y/2*(F_x_f_2*cos(delta_f_2) - F_y_f_2*sin(delta_f_2)) ...
    + l_y/2*(-F_x_r_1*cos(delta_r_1) + F_y_r_1*sin(delta_r_1)) ...
    + l_y/2*(F_x_r_2*cos(delta_r_2) - F_y_r_2*sin(delta_r_2)) ...
    + l_f*(F_x_f_1*sin(delta_f_1) + F_y_f_1*cos(delta_f_1)) ...
    + l_f*(F_x_f_2*sin(delta_f_2) + F_y_f_2*cos(delta_f_2)) ...
    - l_r*(F_x_r_1*sin(delta_r_1) + F_y_r_1*cos(delta_r_1)) ...
    - l_r*(F_x_r_2*sin(delta_r_2) + F_y_r_2*cos(delta_r_2)));

% 输出
D_X = [D_v;D_beta;D_omega];

end