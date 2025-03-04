# pragma once

# define DISP_VAR(x) std::cout<<"\n"<<#x<<" : \n"<<x<<"\n"<<std::endl;

// eigen lib
# include <Eigen/Core>
# include <Eigen/SparseCore>

// self headers
# include "homo_ocp_linAGB.h"

// standard lib
# include <iostream>
# include <functional>
# include <cmath>
# include <limits>
# include <vector>
# include <omp.h>
# include <sched.h>
# include <chrono>
# include <memory>
# include <exception>

// define structures
namespace homo_ocp
{ 
    using casadi_fun = std::function<int(const double**, double**, int*, double*, int)>;
    typedef Eigen::SparseMatrix<double,Eigen::ColMajor> SpMat; 

    // -----------------------------------------------------------------------
    // ---------------------------------定义结构-------------------------------
    // -----------------------------------------------------------------------
    struct clock
    {
        std::chrono::_V2::system_clock::time_point start;
        std::chrono::_V2::system_clock::time_point end;

        clock()
        {
            start = std::chrono::high_resolution_clock::now();
            end = std::chrono::high_resolution_clock::now();
        }

        void tic()
        {
            start = std::chrono::high_resolution_clock::now();
        }

        // 输入：bool disp，是否打印
        double toc(bool disp)
        {
            end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double,std::milli> elapsed = end - start;

            if (disp)
            {
                std::cout << "CPU time = " << elapsed.count() << " ms\n";
            }

            return elapsed.count();
        }

        double toc()
        {
            return toc(1);
        }
    };

    struct symCSC
    {
        int n; // 列（或行）个数

        Eigen::VectorXi ptr;
        Eigen::VectorXi row;
        Eigen::VectorXd val;
    };

    struct prob
    {
        casadi_fun fun_f;
        casadi_fun fun_eq;
        casadi_fun fun_df;
        casadi_fun fun_deq;
        casadi_fun fun_h_lower;

        Eigen::VectorXd p;

        Eigen::VectorXd lbx;
        Eigen::VectorXd ubx;

        int n_x;
        int n_eq;
        int n_p;

        Eigen::VectorXi nz_A;
        Eigen::VectorXi nz_h;
    };

    struct sol
    {
        Eigen::VectorXd x;
        Eigen::VectorXd lambda;
        Eigen::VectorXd s_lbx;
        Eigen::VectorXd s_ubx;
    };

    struct wsInfo
    {
        bool isInit = false;
        sol sol_;
        Eigen::VectorXd p;
    };
    
    struct basicParam
    {
        int iter_max = 1e2; // 最大迭代次数
        double accept_tol = 1e-6; // KKT系统容差，达到容差后认为收敛
        double small_tol = 1e-15; // 无穷小，小于该值认为等于0
        double kappa_1 = 0.01; // 初始值到单边界的距离系数
        double kappa_2 = 0.01; // 初始值到双边界的距离系数
        double lambda_max = 1e4; // 最大等式乘子
        double tau_min = 0.9; // 最小步长因子
        double tau_max = 1 - 1e-6; // 最大步长因子
        double merit_sigma_max = 10; // 最大优势函数罚参
        int soc_iter_max = 2; // 二阶校正最大允许次
        double soc_k = 0.99; // 二阶校正新的等式误差最小下降为上次的soc_k倍，否则退出二阶校正
        int printLevel = 2; // <0: 不打印消息; >=0：打印警告信息；>=1：打印结果消息；>=2：打印过程消息
        bool useHomotopy = false; // 是否使用参数同伦进行温启动
        
        struct monotone
        {
            double k = 0.2; // 罚参数更新参数
            double t = 1.5; // 罚参数更新参数
            double epsilon = 10; // 罚参数更新条件参数
        } monotone;

        struct loqo
        {
            double k = 0.05; // 罚参数更新参数
            double t = 2; // 罚参数更新参数
            double alpha = 0.1; // 罚参数更新参数
            double mu_min = 1e-9; // 最小罚参数
        } loqo;
        
        struct quality
        {
            double epsilon = 0.01; // 与确定区间有关的参数，需要比较小
            int iter_max = 12; // 黄金分割法最大迭代次数
            int block = 6; // 搜索区间划分个数
            double accept_tol = 1e-2; // 黄金分割容许相对误差
            double sigma_max = 1e3;
            double sigma_min = 1e-8;
            double mu_min = 1e-9;
        } quality;
        
        struct lineSearch
        {
            int iter_max = 8; // 线搜索最大允许次数
            double beta = 0.7; // 参数，范围（0，1）
            double eta = 0.1; // 参数，范围（0，1）
        } lineSearch;
    };

    struct ICor
    {
        double w; // 非凸问题数修正系数
        double a; // 0特征值修正系数
        double w_ac = 10; // w的上升速度
        double a_ac = 10; // a的上升速度
        double w_dc = 2; // w的下降速度
        double a_dc = 2; // a的下降速度
        double w_0 = 1e-2; // 初始值
        double a_0 = 1e-6; // 初始值
        double w_min = 1e-4;
        double a_min = 1e-8;
        double w_max = 1e4;
        double a_max = 1e4;
        int switchIter = 3; // 模式切换的迭代次数
        int porbeIter = 3; // 在修正模式下试图回到一般模式的试探频率
        int w_count = 0;
        int w_count_2 = 0;
        bool wCorFlag = 0;
        int a_count = 0;
        int a_count_2 = 0;
        bool aCorFlag = 0;

        ICor()
        {
            w = w_0;
            a = a_0;
        }

        void refresh()
        {
            w = w_0;
            a = a_0;

            w_count = 0;
            w_count_2 = 0;
            wCorFlag = 0;
            
            a_count = 0;
            a_count_2 = 0;
            aCorFlag = 0;
        }
    };

    struct barrier
    {
        enum class updateMode
        {
            Monotone,
            LOQO_rule,
            QualityFunction,
        };

        double mu_max = 1e3; // 最大罚参数
        double mu_0 = 0.1; // 初始罚参数
        double mu; // 当前罚参数
        updateMode updateMode_ = updateMode::QualityFunction;
            // 更新模式，包括：
            // Monotone：单调模式，参考IPOPT
            // LOQO_rule：参考LOQO中的启发式罚参数更新模式
            // Quality_function：以指标函数确定新的罚参数
        bool isProtected = false; // 保护模式，当此模式开启后会将mu按照一定规则重置并进入Monotone模式，
                                // 直到新的迭代被直接接受而不使用线搜索（严格来说，这并不是参数）
        double k = 0.8; // 保护模式开启时重置mu有关的参数

        barrier()
        {
            mu = mu_0;
        }

        void refresh()
        {
            mu = mu_0;
            isProtected = false;
        }
    };

    struct filter
    {
        std::vector<std::pair<double,double>> pairs = 
            {std::make_pair(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())}; // 过滤器存储的目标对（严格来说，这并不是参数）
        double beta = 1e-3; // 过滤器最小参数对距离系数

        bool isAcceptabel(std::pair<double,double> pair_new);

        void refresh()
        {
            pairs = {std::make_pair(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())};
        }
    };

    struct watchDog
    {
        int iter = 0;
        int iter_max = 4;  // 看门狗最大允许非单调次数
        double merit_sigma = 0;
        double PHI_0;
        double D_0;
        std::vector<sol> start;
        std::vector<sol> steps;
        std::vector<double> alpha_primal;
        std::vector<double> alpha_dual;
        std::vector<double> PHI;
        std::pair<double, double> pair_0;
        std::vector<std::pair<double, double>> pairs;

        watchDog() : start(iter_max + 1), steps(iter_max + 1), alpha_primal(iter_max + 1), 
            alpha_dual(iter_max + 1), PHI(iter_max + 1), pairs(iter_max + 1){}

        void refresh()
        {
            iter = 0;
            merit_sigma = 0;
            PHI_0 = 0;
            D_0 = 0;

            std::vector<sol> new_start(iter_max + 1);
            std::vector<sol> new_steps(iter_max + 1);
            std::vector<double> new_alpha_primal(iter_max + 1);
            std::vector<double> new_alpha_dual(iter_max + 1);
            std::vector<double> new_PHI(iter_max + 1);
            std::pair<double, double> new_pair_0;
            std::vector<std::pair<double, double>> new_pairs(iter_max + 1);
            
            start = new_start;
            steps = new_steps;
            alpha_primal = new_alpha_primal;
            alpha_dual = new_alpha_dual;
            PHI = new_PHI;
            pair_0 = new_pair_0;
            pairs = new_pairs;
        }
    };

    struct homotopy
    {
        double lota = 0; // 当前同伦参数，非定植
        double diff_lota_min = 1.0; // 最小同伦步长
        double diff_lota_max = 1.0; // 最大同伦步长
        double epsilon = 1; // 单调更新参数
        int iter_max = 100;
        bool isDone = false;

        void refresh()
        {
            lota = 0;
            isDone = false;
        }
    };

    struct processRecorder
    {
        int n_f = 0;
        double t_f = 0;
        int n_df = 0;
        double t_df = 0;
        int n_h = 0;
        double t_h = 0;
        int n_eq = 0;
        double t_eq = 0;
        int n_deq = 0;
        double t_deq = 0;
        int n_factor = 0;
        double t_factor = 0;
        int n_solve = 0;
        double t_solve = 0;
    };

    // -----------------------------------------------------------------------
    // ---------------------------------定义函数-------------------------------
    // -----------------------------------------------------------------------

    inline void get_symCSC(SpMat& m_SpMat, symCSC& m_symCSC)
    {
        if (!m_SpMat.isCompressed())
        {
            m_SpMat.makeCompressed();
        }
        
        m_symCSC.ptr = Eigen::VectorXi::Zero(m_SpMat.outerSize() + 1);
        Eigen::Map<Eigen::VectorXi> tmp(const_cast<int*>(m_SpMat.outerIndexPtr()), m_SpMat.outerSize());

        m_symCSC.ptr(Eigen::seq(0,Eigen::last - 1)) = tmp;
        m_symCSC.ptr(m_SpMat.outerSize()) = m_SpMat.nonZeros();
        m_symCSC.row = Eigen::VectorXi::Map(m_SpMat.innerIndexPtr(),m_symCSC.ptr(Eigen::last));
        m_symCSC.val = Eigen::VectorXd::Map(m_SpMat.valuePtr(),m_symCSC.ptr(Eigen::last));

        m_symCSC.n = m_SpMat.outerSize();
    }

    // -----------------------------------------------------------------------
    // ---------------------------------定义类---------------------------------
    // -----------------------------------------------------------------------
    
    class solver
    {
    private:
        std::vector<int> idx_lbx;
        std::vector<int> idx_ubx;

        int n_lbx;
        int n_ubx;

        Eigen::VectorXi mapNNZ_A;
        Eigen::VectorXi mapNNZ_h;
        Eigen::VectorXi mapNNZ_KKT_K;

        Eigen::VectorXd diag_Ix;
        Eigen::VectorXd diag_Ieq;

        int cpu_core;
        double inf_;

        prob prob_;
        sol sol_;
        double sol_KKT_error;

        int iter = 0;
        int homotopyConvergeIter = 0;
        double cpuTime = 0;
        int exit_flag = 0;
        bool refresh_flag = true;

        clock clock_;
        processRecorder processRecorder_;

        wsInfo wsInfo_;
        
        void mem_allocate();
        void bound_check();
        
        void get_val_f(const Eigen::VectorXd& x, const Eigen::VectorXd& p, double& val_f, const double k);
        void get_val_df(const Eigen::VectorXd& x, const Eigen::VectorXd& p, Eigen::VectorXd& val_df, const double k);
        void get_val_eq(const Eigen::VectorXd& x, const Eigen::VectorXd& p, Eigen::VectorXd& val_eq, const double k);
        void get_val_deq(const Eigen::VectorXd& x, const Eigen::VectorXd& p, SpMat& KKT_K, const double k);
        void get_val_h(const Eigen::VectorXd& x, const Eigen::VectorXd& lambda, const Eigen::VectorXd& p, SpMat& KKT_K, const double k);

        std::pair<double,double> get_stepLength(
            const Eigen::VectorXd& diff_x, const Eigen::VectorXd& diff_s_lbx, const Eigen::VectorXd& diff_s_ubx, const double tau);
        double get_stepLength_xOnly(const Eigen::VectorXd& diff_x, const double tau);
        double get_stepLength_sOnly(const Eigen::VectorXd& diff_s_lbx, const Eigen::VectorXd& diff_s_ubx, const double tau);
        double quality_function(
            const Eigen::VectorXd& diff_x, const Eigen::VectorXd& diff_s_lbx, const Eigen::VectorXd& diff_s_ubx,
            const Eigen::VectorXd& P1, const Eigen::VectorXd& P2, const Eigen::VectorXd& P3, const std::pair<double,double> alpha);

        int linSolver_init(SpMat&);
        std::pair<bool,int> getInertia(SpMat&);// return {is singular, number of negitive eigen values}
        int KKT_factor(SpMat&);
        int KKT_factor_final(SpMat&);
        int KKT_solve1(const Eigen::VectorXd& KKT_b, Eigen::VectorXd& x);
        int KKT_solve2(const Eigen::VectorXd& KKT_B, Eigen::VectorXd& X);

        void showThread();
        void showSpMat(SpMat&);

        // QP
        bool isQPprob = false;
        SpMat QP_H;
        symCSC QP_H_CSC;
        Eigen::VectorXd QP_c;
        SpMat QP_A;
        symCSC QP_A_CSC;
        Eigen::VectorXd QP_b;

        // =======================尺寸较大的过程变量=======================
        SpMat fullZeroK;
        Eigen::VectorXd KKT_b;

        factorInfo factorInfo_;

        SpMat H;
        SpMat A;

        Eigen::VectorXd v_lbx_active;
        Eigen::VectorXd v_ubx_active;

        Eigen::VectorXd s_lbx_active;
        Eigen::VectorXd s_ubx_active;
        Eigen::VectorXd val_df;

        SpMat tmp_K;
        Eigen::VectorXd tmp_sol;
        symCSC tmp_symCSC;

        Eigen::VectorXd r_x;
        Eigen::VectorXd r_lambda;
        Eigen::VectorXd r_lbxs;
        Eigen::VectorXd r_ubxs;

        Eigen::VectorXd d;
        Eigen::VectorXd d0;

        sol diff;

        Eigen::VectorXd KKT_B;

        Eigen::VectorXd r_lbxs_qf_0;
        Eigen::VectorXd r_ubxs_qf_0;
        Eigen::VectorXd R_lbx_qf_0;
        Eigen::VectorXd R_ubx_qf_0;

        Eigen::VectorXd r_lbxs_qf_1;
        Eigen::VectorXd r_ubxs_qf_1;
        Eigen::VectorXd R_lbx_qf_1;
        Eigen::VectorXd R_ubx_qf_1;

        Eigen::VectorXd diff_qf;

        Eigen::VectorXd diff_x_qf_0;
        Eigen::VectorXd diff_lambda_qf_0;
        Eigen::VectorXd diff_s_lbx_qf_0;
        Eigen::VectorXd diff_s_ubx_qf_0;

        Eigen::VectorXd diff_x_qf_1;
        Eigen::VectorXd diff_lambda_qf_1;
        Eigen::VectorXd diff_s_lbx_qf_1;
        Eigen::VectorXd diff_s_ubx_qf_1;

        Eigen::VectorXd P1_qf_0;
        Eigen::VectorXd P2_qf_0;
        Eigen::VectorXd P3_qf_0;

        Eigen::VectorXd P1_qf_1;
        Eigen::VectorXd P2_qf_1;
        Eigen::VectorXd P3_qf_1;

        Eigen::VectorXd tmp_diff_x;
        Eigen::VectorXd tmp_diff_s_lbx;
        Eigen::VectorXd tmp_diff_s_ubx;

        Eigen::VectorXd tmp_P1;
        Eigen::VectorXd tmp_P2;
        Eigen::VectorXd tmp_P3;

        Eigen::VectorXd diff_x_qf_a;
        Eigen::VectorXd diff_s_lbx_qf_a;
        Eigen::VectorXd diff_s_ubx_qf_a;

        Eigen::VectorXd diff_x_qf_b;
        Eigen::VectorXd diff_s_lbx_qf_b;
        Eigen::VectorXd diff_s_ubx_qf_b;

        Eigen::VectorXd P1_qf_a;
        Eigen::VectorXd P2_qf_a;
        Eigen::VectorXd P3_qf_a;

        Eigen::VectorXd P1_qf_b;
        Eigen::VectorXd P2_qf_b;
        Eigen::VectorXd P3_qf_b;

        Eigen::VectorXd diff_x_qf_p;
        Eigen::VectorXd diff_lambda_qf_p;
        Eigen::VectorXd diff_s_lbx_qf_p;
        Eigen::VectorXd diff_s_ubx_qf_p;

        Eigen::VectorXd diff_x_qf_q;
        Eigen::VectorXd diff_lambda_qf_q;
        Eigen::VectorXd diff_s_lbx_qf_q;
        Eigen::VectorXd diff_s_ubx_qf_q;

        Eigen::VectorXd P1_qf_p;
        Eigen::VectorXd P2_qf_p;
        Eigen::VectorXd P3_qf_p;

        Eigen::VectorXd P1_qf_q;
        Eigen::VectorXd P2_qf_q;
        Eigen::VectorXd P3_qf_q;

        Eigen::VectorXd r_lbxs_relaxed;
        Eigen::VectorXd r_ubxs_relaxed;
        Eigen::VectorXd R_lbx;
        Eigen::VectorXd R_ubx;

        Eigen::VectorXd x_new;
        Eigen::VectorXd val_eq_new;
        Eigen::VectorXd val_eq;

        Eigen::VectorXd r_lambda_soc;
        Eigen::VectorXd KKT_b_soc;
        Eigen::VectorXd diff_x_soc;
        Eigen::VectorXd val_eq_soc_new;
        Eigen::VectorXd tmp_x;
        Eigen::VectorXd x_soc;
        Eigen::VectorXd val_eq_soc;
        Eigen::VectorXd tmp_val_eq;

        Eigen::VectorXd tmp_nlbx;
        Eigen::VectorXd tmp_nubx;
        // ============================================================
    public:
        /// @brief 一般非线性规划问题接口
        /// @param prob
        /// @param what_is_inf 
        /// @param cpu_core 
        /// @param printLevel 
        explicit solver(prob, const double what_is_inf, const int cpu_core, const int printLevel);
        
        /// @brief 二次型问题的接口
        /// @param H 
        /// @param c 
        /// @param A_eq 
        /// @param b_eq 
        /// @param A_ineq 
        /// @param b_ineq 
        /// @param lbx 
        /// @param ubx 
        explicit solver(
            SpMat &H, Eigen::VectorXd &c, 
            SpMat &A_eq, Eigen::VectorXd &b_eq, 
            SpMat &A_ineq, Eigen::VectorXd &b_ineq, 
            Eigen::VectorXd &lbx, Eigen::VectorXd &ubx,
            const double what_is_inf, const int cpu_core, const int printLevel
        );

        ~solver();

        void refresh(); // 刷新过程量
        void reLoadParam(const Eigen::VectorXd& p); // 修改问题的参数
        void reLoadBoundary(const Eigen::VectorXd *lbx, const Eigen::VectorXd *ubx, const bool check); //修改问题的边界
        void setWsInfo(const sol &m_sol, const  Eigen::VectorXd &p); // 使用其他已知的温启动信息
        void clearWsInfo(); // 将同伦温启动信息清除

        void reLoadQP(const Eigen::VectorXd *c, const Eigen::VectorXd *b_eq, const Eigen::VectorXd *b_ineq);// 快速变换QP问题的部分参数
        void reLoadQPboundary(const Eigen::VectorXd *lbx, const Eigen::VectorXd *ubx, const bool check);

        sol solve(const Eigen::VectorXd& x0);

        void showProcess();
        const prob* getProb();
        const sol* getSolution();
        int getIteration();
        double getCpuTime();
        int getHomotopyConvergeIteration();
        const processRecorder* getProcessInfo();
        int getExitFlag();
        const wsInfo* getWsInfo();

        void setInfinity(double arg1){
            inf_ = arg1;
        };

        basicParam basicParam_;
        ICor ICor_;
        barrier barrier_;
        filter filter_;
        watchDog watchDog_;
        homotopy homotopy_;

        solver(const solver&) = delete;
        solver& operator = (const solver&) = delete;
    };
}; // namespace homo_ocp

# ifndef HOMO_OCP_PROB
# define HOMO_OCP_PROB(PROB_NAME,FUN_NAME) \
    PROB_NAME.fun_f = FUN_NAME##_f; \
    PROB_NAME.fun_df = FUN_NAME##_df; \
    PROB_NAME.fun_eq = FUN_NAME##_eq; \
    PROB_NAME.fun_deq = FUN_NAME##_deq; \
    PROB_NAME.fun_h_lower = FUN_NAME##_h_lower; \
    PROB_NAME.n_x = FUN_NAME##_h_lower_sparsity_in(0)[3]; \
    PROB_NAME.n_eq = FUN_NAME##_h_lower_sparsity_in(1)[3]; \
    PROB_NAME.n_p = FUN_NAME##_h_lower_sparsity_in(2)[3];\
    const int* nz_A = FUN_NAME##_deq_sparsity_out(0); \
    const int* nz_h = FUN_NAME##_h_lower_sparsity_out(0); \
    PROB_NAME.nz_A = Eigen::VectorXi::Map(nz_A,3 + nz_A[1] + nz_A[2 + nz_A[1]]); \
    PROB_NAME.nz_h = Eigen::VectorXi::Map(nz_h,3 + nz_h[1] + nz_h[2 + nz_h[1]]); \
    PROB_NAME.p.resize(PROB_NAME.n_p); \
    PROB_NAME.lbx.resize(PROB_NAME.n_x); \
    PROB_NAME.ubx.resize(PROB_NAME.n_x);
# endif