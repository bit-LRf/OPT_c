# include "homo_ocp.h"

namespace homo_ocp
{
    // 求解其初始化
    solver::solver(prob arg1, const double arg2, const int arg3, const int arg4) :
        prob_(arg1), 
        inf_(arg2), 
        cpu_core(arg3)
    {
        basicParam_.printLevel = arg4;

        // 一些提醒
        // if (basicParam_.printLevel >= 1)
        // {
        //     std::cout<<"|--------------------------------------------------------------------------------------|"<<std::endl;
        //     std::cout<<"|  Please note that only the lower part of hessian should be given from \'fun_h_lower\'  |"<<std::endl;
        //     std::cout<<"|--------------------------------------------------------------------------------------|\n"<<std::endl;
        // }
        
        if (basicParam_.printLevel >= 1)
        {
            printf("====================  start init  ====================\n");

            printf("cpu core setting is: %d\n",cpu_core);
        }

        // 并行初始化
        Eigen::initParallel();
        omp_set_num_threads(cpu_core);

        if (basicParam_.printLevel >= 1)
        {
            printf("set parallel success\n");
        }
        
        // 处理约束
        bound_check();

        // 测量非零元
        # pragma omp parallel sections
        {
            # pragma omp section
            {
                mapNNZ_A = prob_.nz_A(Eigen::seq(3,3 + prob_.n_x - 1)) - prob_.nz_A(Eigen::seq(2,2 + prob_.n_x - 1));

                if (basicParam_.printLevel >= 1)
                {
                    printf("nonZero elements of deq: %d\n",prob_.nz_A(3 + prob_.n_x - 1));
                }
            }

            # pragma omp section
            {
                mapNNZ_h = prob_.nz_h(Eigen::seq(3,3 + prob_.n_x - 1)) - prob_.nz_h(Eigen::seq(2,2 + prob_.n_x - 1));

                if (basicParam_.printLevel >= 1)
                {
                    printf("nonZero elements of hessian: %d\n",prob_.nz_h(3 + prob_.n_x - 1));
                }
            }
        }

        mapNNZ_KKT_K = Eigen::VectorXi::Ones(prob_.n_x + prob_.n_eq);
        mapNNZ_KKT_K(Eigen::seq(0,prob_.n_x - 1)) = mapNNZ_A + mapNNZ_h;

        // 准备两个用来加的主对角元单位向量
        diag_Ix = Eigen::VectorXd::Zero(prob_.n_x + prob_.n_eq);
        diag_Ix(Eigen::seq(0,prob_.n_x - 1)) = Eigen::VectorXd::Ones(prob_.n_x);
        diag_Ieq = Eigen::VectorXd::Zero(prob_.n_x + prob_.n_eq);
        diag_Ieq(Eigen::lastN(prob_.n_eq)) = Eigen::VectorXd::Ones(prob_.n_eq);

        // linSolver初始化
        Eigen::VectorXd x = Eigen::VectorXd::Zero(prob_.n_x);
        Eigen::VectorXd lambda = Eigen::VectorXd::Zero(prob_.n_eq);
        fullZeroK = SpMat(prob_.n_x + prob_.n_eq,prob_.n_x + prob_.n_eq);
        fullZeroK.reserve(mapNNZ_KKT_K);
        get_val_h(x,lambda,prob_.p,fullZeroK,0);
        get_val_deq(x,prob_.p,fullZeroK,0);
        fullZeroK += 0*diag_Ix.asDiagonal();
        fullZeroK += 0*diag_Ieq.asDiagonal();
        fullZeroK.makeCompressed();
        linSolver_init(fullZeroK);

        // 过程变量内存分配
        mem_allocate();

        // 刷新一下
        refresh();

        if (basicParam_.printLevel >= 1)
        {
            std::cout<<"==================== init success ====================\n"<<std::endl;
        }
    }

    // 刷新一些结构体
    void solver::refresh()
    {
        iter = 0;
        homotopyConvergeIter = 0;
        cpuTime = 0;
        exit_flag = 0;
        refresh_flag = true;

        ICor_.refresh();
        barrier_.refresh();
        filter_.refresh();
        watchDog_.refresh();
        homotopy_.refresh();

        processRecorder new_processRecorder;
        processRecorder_ = new_processRecorder;
    }

    // 重新加载问题的参数
    void solver::reLoadParam(const Eigen::VectorXd& p)
    {
        if (p.size() != prob_.p.size())
        {
            if (basicParam_.printLevel >= 0)
            {
                printf("homo_ocp--warning: worng size of param, n_p = %d\n",prob_.n_p);
            }
        }
        else
        {
            prob_.p = p;
        }
    }

    // 更新问题的边界，使用后若温启动信息不满足约束条件则会被清空
    void solver::reLoadBoundary(const Eigen::VectorXd *lbx, const Eigen::VectorXd *ubx, const bool check)
    {
        if (isQPprob)
        {
            reLoadQPboundary(lbx,ubx,check);
        }
        else
        {
            if (lbx)
            {
                prob_.lbx = *lbx;
            }
            
            if (ubx)
            {
                prob_.ubx = *ubx;
            }

            bool flag = true;
            if (wsInfo_.isInit)
            {
                # pragma omp parallel for
                for (size_t i = 0; i < prob_.n_x; i++)
                {
                    if (wsInfo_.sol_.x(i) <= prob_.lbx(i) || wsInfo_.sol_.x(i) >= prob_.ubx(i))
                    {
                        flag = false;
                    }
                }

                if (!flag)
                {
                    if (basicParam_.printLevel >= 0)
                    {
                        printf("homo_ocp--warning: warm start information has been cleared since new bounds make the solution infeasible");
                    }

                    clearWsInfo();
                }
            }
            
            if (check)
            {
                bound_check();
            }
        }
    }

    // 使用其他已知的温启动信息
    void solver::setWsInfo(const sol &m_sol, const Eigen::VectorXd &p)
    {
        if (m_sol.x.size() != prob_.n_x || m_sol.lambda.size() != prob_.n_eq)
        {
            if (basicParam_.printLevel >= 0)
            {
                printf("homo_ocp--warning: size of warm start information is not consistent, therfor warm start information is not accepted\n");
            }

            return;
        }

        bool flag = true;
        # pragma omp parallel for
        for (size_t i = 0; i < prob_.n_x; i++)
        {
            if (m_sol.x(i) <= prob_.lbx(i) || m_sol.x(i) >= prob_.ubx(i))
            {
                flag = false;
            }
        }

        if (!flag)
        {
            if (basicParam_.printLevel >= 0)
            {
                printf("homo_ocp--warning: warm start information is not accepted since solution is infeasible\n");
            }

            return;
        }
        
        wsInfo_.sol_ = m_sol;
        wsInfo_.p = p;
        wsInfo_.isInit = true;
    }

    // 将同伦温启动信息清除
    void solver::clearWsInfo()
    {
        wsInfo_.isInit = false;
    }
    
    // 检查有效约束
    void solver::bound_check()
    {
        if ((prob_.lbx.array() >= prob_.ubx.array()).count() > 0)
        {
            if (basicParam_.printLevel >= 0)
            {
                std::cerr<<"homo_ocp--warning: some lower bounds is equal or bigger than upper bounds, ";
                std::cerr<<"homo_ocp does not support defining equality constraints by setting the upper and lower bounds equal"<<std::endl;
            }
        }

        if (inf_ < 1)
        {
            if (basicParam_.printLevel >= 0)
            {
                std::cerr<<"homo_ocp--warning: it appears that you set a vary small infinity (inf = "<<inf_<<")"<<std::endl;
            }
        }
        
        # pragma omp parallel sections
        {
            # pragma omp section 
            {
                int tmp_ptr = 0;

                n_lbx = (abs(prob_.lbx.array()) < inf_).count();
                idx_lbx.resize(n_lbx);

                for (int i = 0; i < prob_.lbx.size(); i++)
                {
                    if (abs(prob_.lbx[i]) < inf_)
                    {
                        idx_lbx[tmp_ptr] = i;

                        tmp_ptr += 1;
                    }
                }
            }

            # pragma omp section
            {
                int tmp_ptr = 0;

                n_ubx = (abs(prob_.ubx.array()) < inf_).count();
                idx_ubx.resize(n_ubx);

                for (int i = 0; i < prob_.ubx.size(); i++)
                {
                    if (abs(prob_.ubx[i]) < inf_)
                    {
                        idx_ubx[tmp_ptr] = i;

                        tmp_ptr += 1;
                    }
                }
            }
        }

        if (basicParam_.printLevel >= 1)
        {
            printf("valid boundary: n_lbx = %d, n_ubx = %d \n",n_lbx,n_ubx);
        }
    }

    // 向量矩阵初始化
    void solver::mem_allocate()
    {
        H.resize(prob_.n_x,prob_.n_x);
        A.resize(prob_.n_eq,prob_.n_x);

        v_lbx_active.resize(n_lbx);
        v_ubx_active.resize(n_ubx);
        s_lbx_active.resize(n_lbx);
        s_ubx_active.resize(n_ubx);
        val_df.resize(prob_.n_x);

        KKT_b.resize(prob_.n_x + prob_.n_eq);
        tmp_K.resize(prob_.n_x + prob_.n_eq,prob_.n_x + prob_.n_eq);
        tmp_sol.resize(prob_.n_x + prob_.n_eq);

        r_x.resize(prob_.n_x);
        r_lambda.resize(prob_.n_eq);
        r_lbxs.resize(n_lbx);
        r_ubxs.resize(n_ubx);

        d.resize(prob_.n_x);
        d0.resize(prob_.n_x + prob_.n_eq);

        diff.x.resize(prob_.n_x);
        diff.lambda.resize(prob_.n_eq);
        diff.s_lbx.resize(prob_.n_x);
        diff.s_ubx.resize(prob_.n_x);

        KKT_B.resize(2*(prob_.n_x + prob_.n_eq));

        r_lbxs_qf_0.resize(n_lbx);
        r_ubxs_qf_0.resize(n_ubx);
        R_lbx_qf_0.resize(prob_.n_x);
        R_ubx_qf_0.resize(prob_.n_x);

        r_lbxs_qf_1.resize(n_lbx);
        r_ubxs_qf_1.resize(n_ubx);
        R_lbx_qf_1.resize(prob_.n_x);
        R_ubx_qf_1.resize(prob_.n_x);

        diff_qf.resize(2*(prob_.n_x + prob_.n_eq));

        diff_x_qf_0.resize(prob_.n_x);
        diff_lambda_qf_0.resize(prob_.n_eq);
        diff_s_lbx_qf_0.resize(prob_.n_x);
        diff_s_ubx_qf_0.resize(prob_.n_x);

        diff_x_qf_1.resize(prob_.n_x);
        diff_lambda_qf_1.resize(prob_.n_eq);
        diff_s_lbx_qf_1.resize(prob_.n_x);
        diff_s_ubx_qf_1.resize(prob_.n_x);

        P1_qf_0.resize(prob_.n_x);
        P2_qf_0.resize(prob_.n_eq);
        P3_qf_0.resize(prob_.n_x);

        P1_qf_1.resize(prob_.n_x);
        P2_qf_1.resize(prob_.n_eq);
        P3_qf_1.resize(prob_.n_x);

        tmp_diff_x.resize(prob_.n_x);
        tmp_diff_s_lbx.resize(prob_.n_x);
        tmp_diff_s_ubx.resize(prob_.n_x);

        tmp_P1.resize(prob_.n_x);
        tmp_P2.resize(prob_.n_eq);
        tmp_P3.resize(prob_.n_x);

        diff_x_qf_a.resize(prob_.n_x);
        diff_s_lbx_qf_a.resize(prob_.n_x);
        diff_s_ubx_qf_a.resize(prob_.n_x);

        diff_x_qf_b.resize(prob_.n_x);
        diff_s_lbx_qf_b.resize(prob_.n_x);
        diff_s_ubx_qf_b.resize(prob_.n_x);

        P1_qf_a.resize(prob_.n_x);
        P2_qf_a.resize(prob_.n_eq);
        P3_qf_a.resize(prob_.n_x);

        P1_qf_b.resize(prob_.n_x);
        P2_qf_b.resize(prob_.n_eq);
        P3_qf_b.resize(prob_.n_x);

        r_lbxs_relaxed.resize(n_lbx);
        r_ubxs_relaxed.resize(n_ubx);
        R_lbx.resize(prob_.n_x);
        R_ubx.resize(prob_.n_x);

        x_new.resize(prob_.n_x);
        val_eq_new.resize(prob_.n_eq);
        val_eq.resize(prob_.n_eq);

        r_lambda_soc.resize(prob_.n_eq);
        KKT_b_soc.resize(prob_.n_x + prob_.n_eq);
        diff_x_soc.resize(prob_.n_x);
        val_eq_soc_new.resize(prob_.n_eq);
        tmp_x.resize(prob_.n_x);
        x_soc.resize(prob_.n_x);
        val_eq_soc.resize(prob_.n_eq);
        tmp_val_eq.resize(prob_.n_eq);

        tmp_nlbx.resize(n_lbx);
        tmp_nubx.resize(n_ubx);
    }

    solver::~solver()
    {

    }
} // namespace homo_ocp
