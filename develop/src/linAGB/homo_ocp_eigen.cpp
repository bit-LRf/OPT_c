# include "homo_ocp.h"

# ifdef HOMO_OCP_USE_EIGEN
namespace homo_ocp
{
    int solver::linSolver_init(SpMat& KKT_KI)
    {
        if (basicParam_.printLevel >= 0)
        {
            std::cerr<<"Warning: this linear solver will not cheack your problem's convexity"<<std::endl;
        }

        return 0;
    }

    std::pair<bool,int> solver::getInertia(SpMat& KKT_K)
    {
        return {KKT_factor(KKT_K) != 0,prob_.n_eq};
    }

    int solver::KKT_factor(SpMat& KKT_K)
    {
        clock_.tic();

        factorInfo_.linSolver.compute(KKT_K.selfadjointView<Eigen::Lower>());

        processRecorder_.n_factor += 1;
        processRecorder_.t_factor += clock_.toc(0);

        if (factorInfo_.linSolver.info() != 0)
        {
            return -1;
        }

        return 0;
    }

    int solver::KKT_factor_final(SpMat& KKT_K)
    {
        return 0;
    }

    int solver::KKT_solve1(const Eigen::VectorXd& KKT_b, Eigen::VectorXd& x)
    {
        clock_.tic();

        x = factorInfo_.linSolver.solve(KKT_b);

        processRecorder_.n_solve += 1;
        processRecorder_.t_solve += clock_.toc(0);

        return 0;
    }

    // 求解两个线性方程组，x = [x1;x2]
    int solver::KKT_solve2(const Eigen::VectorXd& KKT_B, Eigen::VectorXd& x)
    {
        clock_.tic();
        
        int n = x.size()/2;

        x(Eigen::seq(0,n - 1)) = factorInfo_.linSolver.solve(KKT_B(Eigen::seq(0,n - 1)));
        x(Eigen::lastN(n)) = factorInfo_.linSolver.solve(KKT_B(Eigen::lastN(n)));

        processRecorder_.n_solve += 2;
        processRecorder_.t_solve += clock_.toc(0);

        return 0;
    }
};
# endif