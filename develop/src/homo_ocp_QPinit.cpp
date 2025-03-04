# include "homo_ocp.h"

namespace homo_ocp
{
    solver::solver(
        SpMat &H, Eigen::VectorXd &c, 
        SpMat &A_eq, Eigen::VectorXd &b_eq, 
        SpMat &A_ineq, Eigen::VectorXd &b_ineq, 
        Eigen::VectorXd &lbx, Eigen::VectorXd &ubx,
        const double what_is_inf, const int cpu_core, const int printLevel
    ):
        inf_(what_is_inf), 
        cpu_core(cpu_core),
        isQPprob(true)
    {
        basicParam_.printLevel = printLevel;

        SpMat H_tril = H.triangularView<Eigen::Lower>();
        H_tril.makeCompressed();

        if (basicParam_.printLevel >= 2)
        {
            printf("====================  start init  ====================\n");
        }

        // 并行初始化
        Eigen::initParallel();
        omp_set_num_threads(cpu_core);

        if (basicParam_.printLevel >= 2)
        {
            printf("set parallel success\n");
        }

        // 松弛问题，检查非零元
        prob_.n_x = c.size() + b_ineq.size();
        prob_.n_eq = b_eq.size() + b_ineq.size();
        QP_H.resize(prob_.n_x,prob_.n_x);
        QP_A.resize(prob_.n_eq,prob_.n_x);

        get_symCSC(H_tril,tmp_symCSC);
        mapNNZ_h.resize(prob_.n_x);
        mapNNZ_h.setZero();
        mapNNZ_h(Eigen::seq(0,c.size() - 1)) += tmp_symCSC.ptr(Eigen::seq(1,c.size())) - tmp_symCSC.ptr(Eigen::seq(0,c.size() - 1));
        QP_H.reserve(mapNNZ_h);
        # pragma omp parallel for
        for (size_t i = 0; i < tmp_symCSC.ptr.size(); i++)
        {
            for (size_t j = 0; j < mapNNZ_h(i); j++)
            {
                int nz = tmp_symCSC.ptr(i) + j;
                int row = tmp_symCSC.row(nz);
                QP_H.coeffRef(row,i) += H_tril.coeffRef(row,i);
            }
        }
        
        if (basicParam_.printLevel >= 1)
        {
            std::cout<<"variables: "<<c.size()<<"\nslack variables: "<<b_ineq.size()<<std::endl;
        }
        
        get_symCSC(A_eq,tmp_symCSC);
        Eigen::VectorXi mapNNZ_eq;
        mapNNZ_eq.resize(prob_.n_x);
        mapNNZ_eq.setZero();
        if (b_eq.size() != 0)
        {
            mapNNZ_eq(Eigen::seq(0,c.size() - 1)) = tmp_symCSC.ptr(Eigen::seq(1,c.size())) - tmp_symCSC.ptr(Eigen::seq(0,c.size() - 1));
            QP_A.reserve(mapNNZ_eq);
            # pragma omp parallel for
            for (size_t i = 0; i < tmp_symCSC.ptr.size(); i++)
            {
                for (size_t j = 0; j < mapNNZ_eq(i); j++)
                {
                    int nz = tmp_symCSC.ptr(i) + j;
                    int row = tmp_symCSC.row(nz);
                    QP_A.coeffRef(row,i) += A_eq.coeffRef(row,i);
                }
            }
        }

        get_symCSC(A_ineq,tmp_symCSC);
        Eigen::VectorXi mapNNZ_ineq;
        mapNNZ_ineq.resize(prob_.n_x);
        mapNNZ_ineq.setZero();
        if (b_ineq.size() != 0)
        {
            mapNNZ_ineq(Eigen::seq(0,c.size() - 1)) = tmp_symCSC.ptr(Eigen::seq(1,c.size())) - tmp_symCSC.ptr(Eigen::seq(0,c.size() - 1));
            QP_A.reserve(mapNNZ_ineq);
            # pragma omp parallel for
            for (size_t i = 0; i < tmp_symCSC.ptr.size(); i++)
            {
                for (size_t j = 0; j < mapNNZ_ineq(i); j++)
                {
                    int nz = tmp_symCSC.ptr(i) + j;
                    int row = tmp_symCSC.row(nz);
                    QP_A.coeffRef(row + b_eq.size(),i) += A_ineq.coeffRef(row,i);
                }
            }
        }
        
        mapNNZ_A = Eigen::VectorXi::Ones(prob_.n_x);
        mapNNZ_A(Eigen::seq(0,c.size() - 1)) = Eigen::VectorXi::Zero(c.size());
        QP_A.reserve(mapNNZ_A);
        # pragma omp parallel for
        for (size_t i = b_eq.size(); i < b_eq.size() + b_ineq.size(); i++)
        {
            int col = i - b_eq.size() + c.size();
            QP_A.coeffRef(i,col) += -1;
        }
        mapNNZ_A += mapNNZ_eq + mapNNZ_ineq;

        QP_c.resize(prob_.n_x);
        QP_b.resize(prob_.n_eq);
        QP_c.topRows(c.size()) = c;
        QP_b.topRows(b_eq.size()) = b_eq;
        QP_b.bottomRows(b_ineq.size()) = b_ineq;

        prob_.lbx.resize(prob_.n_x);
        prob_.lbx.topRows(lbx.size()) = lbx;
        prob_.lbx.bottomRows(b_ineq.size()) = Eigen::VectorXd::Zero(b_ineq.size());

        prob_.ubx.resize(prob_.n_x);
        prob_.ubx.topRows(ubx.size()) = ubx;
        prob_.ubx.bottomRows(b_ineq.size()) = Eigen::VectorXd::Constant(b_ineq.size(),inf_);

        mapNNZ_KKT_K = Eigen::VectorXi::Ones(prob_.n_x + prob_.n_eq);
        mapNNZ_KKT_K(Eigen::seq(0,prob_.n_x - 1)) = mapNNZ_A + mapNNZ_h;

        get_symCSC(QP_H,QP_H_CSC);
        prob_.nz_h.resize(2 + QP_H_CSC.ptr.size() + QP_H_CSC.row.size());
        prob_.nz_h << QP_H.rows(), QP_H.cols(), QP_H_CSC.ptr, QP_H_CSC.row;

        get_symCSC(QP_A,QP_A_CSC);
        prob_.nz_A.resize(2 + QP_A_CSC.ptr.size() + QP_A_CSC.row.size());
        prob_.nz_A << QP_A.rows(), QP_A.cols(), QP_A_CSC.ptr, QP_A_CSC.row;

        // 处理约束
        bound_check();

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

        if (basicParam_.printLevel >= 2)
        {
            std::cout<<"==================== init success ====================\n"<<std::endl;
        }
    }

    // 重新加载QP的部分参数，传入空指针代表不更新
    void solver::reLoadQP(const Eigen::VectorXd *c, const Eigen::VectorXd *b_eq, const Eigen::VectorXd *b_ineq)
    {
        if (c)
        {
            QP_c.topRows(c->size()) = *c;
        }

        if (b_eq)
        {
            QP_b.topRows(b_eq->size()) = *b_eq;
        }
        
        if (b_ineq)
        {
            QP_b.bottomRows(b_ineq->size()) = *b_ineq;
        }
    }

    // 更新问题的边界，使用后若温启动信息不满足约束条件则会被清空
    void solver::reLoadQPboundary(const Eigen::VectorXd *lbx, const Eigen::VectorXd *ubx, const bool check)
    {
        if (lbx)
        {
            prob_.lbx.topRows(lbx->size()) = *lbx;
        }

        if (ubx)
        {
            prob_.ubx.topRows(ubx->size()) = *ubx;
        }

        bool flag = true;
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
            printf("由于边界条件替换导致违反不等式约束，温启动信息清除");

            clearWsInfo();
        }

        if (check)
        {
            bound_check();
        }
    }
} // namespace homo_ocp
