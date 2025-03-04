# include "homo_ocp.h"

namespace homo_ocp
{
    // val_f = k*fun_f(x,p)
    void solver::get_val_f(const Eigen::VectorXd& x, const Eigen::VectorXd& p, double& val_f, const double k)
    {
        clock_.tic();

        if (isQPprob)
        {
            val_f = (0.5*x.transpose()*QP_H*x + QP_c.transpose()*x)(0);
        }
        else
        {
            double* mem_f = new double[1]();

            const double* arg[2];
            arg[0] = x.data();
            arg[1] = p.data();
            double** res = &mem_f;

            prob_.fun_f(arg,res,nullptr,nullptr,0);

            if (k == 0)
            {
                val_f = 0;
            }
            else
            {
                val_f = k*res[0][0];
            }

            delete[] mem_f;
        }

        processRecorder_.n_f += 1;
        processRecorder_.t_f += clock_.toc(0);
    }

    // val_df = k*fun_df(x,p)
    void solver::get_val_df(const Eigen::VectorXd& x, const Eigen::VectorXd& p, Eigen::VectorXd& val_df, const double k)
    {
        clock_.tic();

        if (isQPprob)
        {
            val_df = (QP_H*x + QP_c).transpose();
        }
        else
        {
            double* mem_df = new double[prob_.n_x]();

            const double* arg[2];
            arg[0] = x.data();
            arg[1] = p.data();
            double** res = &mem_df;

            prob_.fun_df(arg,res,nullptr,nullptr,0);

            if (k == 0)
            {
                val_df = Eigen::VectorXd::Zero(prob_.n_x);
            }
            else
            {
                val_df = k*Eigen::VectorXd::Map(mem_df, prob_.n_x);
            }

            delete[] mem_df;
        }

        processRecorder_.n_df += 1;
        processRecorder_.t_df += clock_.toc(0);
    }

    // val_eq = k*fun_eq(x,p)
    void solver::get_val_eq(const Eigen::VectorXd& x, const Eigen::VectorXd& p, Eigen::VectorXd& val_eq, const double k)
    {
        clock_.tic();

        if (isQPprob)
        {
            val_eq = QP_A*x - QP_b;
        }
        else
        {
            double* mem_eq = new double[prob_.n_eq]();

            const double* arg[2];
            arg[0] = x.data();
            arg[1] = p.data();
            double** res = &mem_eq;

            prob_.fun_eq(arg,res,nullptr,nullptr,0);

            if (k == 0)
            {
                val_eq = Eigen::VectorXd::Zero(prob_.n_eq);
            }
            else
            {
                val_eq = k*Eigen::VectorXd::Map(mem_eq, prob_.n_eq);
            }

            delete[] mem_eq;
        }
        
        processRecorder_.n_eq += 1;
        processRecorder_.t_eq += clock_.toc(0);
    }

    // KKT_K的左下角赋值为k*fun_deq(x,p)
    void solver::get_val_deq(const Eigen::VectorXd& x, const Eigen::VectorXd& p, SpMat& KKT_K, const double k)
    {
        clock_.tic();

        if (isQPprob)
        {
            # pragma omp parallel for
            for (size_t i = 0; i < prob_.n_x; i++)
            {
                for (size_t j = 0; j < mapNNZ_A(i); j++)
                {
                    int nz = prob_.nz_A(2 + i) + j;
                    int row = prob_.nz_A(3 + prob_.n_x + nz) + prob_.n_x;

                    KKT_K.coeffRef(row,i) += k*QP_A_CSC.val(nz);
                }
            }
        }
        else
        {
            double* mem_deq = new double[prob_.nz_A(2 + prob_.n_x)]();

            const double* arg[2];
            arg[0] = x.data();
            arg[1] = p.data();
            double** res = &mem_deq;

            prob_.fun_deq(arg,res,nullptr,nullptr,0);
            
            if (k == 0)
            {
                # pragma omp parallel for
                for (size_t i = 0; i < prob_.n_x; i++)
                {
                    for (size_t j = 0; j < mapNNZ_A(i); j++)
                    {
                        int nz = prob_.nz_A(2 + i) + j;
                        int row = prob_.nz_A(3 + prob_.n_x + nz) + prob_.n_x;

                        KKT_K.coeffRef(row,i) += 0;
                    }
                }
            }
            else
            {
                # pragma omp parallel for
                for (size_t i = 0; i < prob_.n_x; i++)
                {
                    for (size_t j = 0; j < mapNNZ_A(i); j++)
                    {
                        int nz = prob_.nz_A(2 + i) + j;
                        int row = prob_.nz_A(3 + prob_.n_x + nz) + prob_.n_x;

                        KKT_K.coeffRef(row,i) += k*res[0][nz];
                    }
                }
            }

            delete[] mem_deq;
        }

        processRecorder_.n_deq += 1;
        processRecorder_.t_deq += clock_.toc(0);
    }

    // KKT的左上角的下三角赋值为k*fun_h_lower(x,lambda,p)
    void solver::get_val_h(const Eigen::VectorXd& x, const Eigen::VectorXd& lambda, const Eigen::VectorXd& p, SpMat& KKT_K, const double k)
    {
        clock_.tic();

        if (isQPprob)
        {
            # pragma omp parallel for
            for (size_t i = 0; i < prob_.n_x; i++)
            {
                for (size_t j = 0; j < mapNNZ_h(i); j++)
                {
                    int nz = prob_.nz_h(2 + i) + j;
                    int row = prob_.nz_h(3 + prob_.n_x + nz);

                    KKT_K.coeffRef(row,i) += k*QP_H_CSC.val(nz);
                }
            }
        }
        else
        {
            double* mem_h = new double[prob_.nz_h(2 + prob_.n_x)]();

            const double* arg[3];
            arg[0] = x.data();
            arg[1] = lambda.data();
            arg[2] = p.data();
            double** res = &mem_h;

            prob_.fun_h_lower(arg,res,nullptr,nullptr,0);

            if (k == 0)
            {
                # pragma omp parallel for
                for (size_t i = 0; i < prob_.n_x; i++)
                {
                    for (size_t j = 0; j < mapNNZ_h(i); j++)
                    {
                        int nz = prob_.nz_h(2 + i) + j;
                        int row = prob_.nz_h(3 + prob_.n_x + nz);

                        KKT_K.coeffRef(row,i) += 0;
                    }
                }
            }
            else
            {
                # pragma omp parallel for
                for (size_t i = 0; i < prob_.n_x; i++)
                {
                    for (size_t j = 0; j < mapNNZ_h(i); j++)
                    {
                        int nz = prob_.nz_h(2 + i) + j;
                        int row = prob_.nz_h(3 + prob_.n_x + nz);

                        KKT_K.coeffRef(row,i) += k*res[0][nz];
                    }
                }
            }
            
            delete[] mem_h;
        }
        
        processRecorder_.n_h += 1;
        processRecorder_.t_h += clock_.toc(0);
    }
} // namespace homo_ocp
