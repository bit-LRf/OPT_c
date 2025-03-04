# include "homo_ocp.h"

# ifdef HOMO_OCP_USE_SPRAL
namespace homo_ocp
{
    int solver::linSolver_init(SpMat& KKT_KI)
    {
        get_symCSC(KKT_KI,tmp_symCSC);
        factorInfo_.ptr = tmp_symCSC.ptr;
        factorInfo_.row = tmp_symCSC.row;
        factorInfo_.options.small = basicParam_.small_tol;

        spral_ssids_analyse_ptr32(
            true, 
            prob_.n_x + prob_.n_eq, 
            NULL, 
            factorInfo_.ptr.data(), 
            factorInfo_.row.data(),
            NULL, 
            &factorInfo_.akeep, 
            &factorInfo_.options,
            &factorInfo_.inform
        );
        if(factorInfo_.inform.flag < 0) {
            printf("spral error as flag %d\n",factorInfo_.inform.flag);
            exit(1);
        }
        else if (factorInfo_.inform.flag == 6)
        {
            printf("warning: your problem is structurally singular\n");
        }

        // 
        if (basicParam_.printLevel >= 2)
        {
            printf("spral analysis is done\n");
        }

        return factorInfo_.inform.flag;
    }

    std::pair<bool,int> solver::getInertia(SpMat& KKT_K)
    {
        int flag = KKT_factor(KKT_K);
        return {flag == -5 || flag == 6 || flag == 7,factorInfo_.inform.num_neg};
    }

    int solver::KKT_factor(SpMat& KKT_K)
    {
        clock_.tic();

        get_symCSC(KKT_K,tmp_symCSC);

        spral_ssids_factor_ptr32(
            false, 
            tmp_symCSC.ptr.data(), 
            tmp_symCSC.row.data(), 
            tmp_symCSC.val.data(), 
            NULL,
            factorInfo_.akeep, 
            &factorInfo_.fkeep, 
            &factorInfo_.options, 
            &factorInfo_.inform
        );
        if(factorInfo_.inform.flag < 0 && factorInfo_.inform.flag != -5) {
            printf("spral error as flag %d\n",factorInfo_.inform.flag);
            exit(1);
        }

        factorInfo_.n = tmp_symCSC.n;

        processRecorder_.n_factor += 1;
        processRecorder_.t_factor += clock_.toc(0);

        return factorInfo_.inform.flag;
    }

    int solver::KKT_factor_final(SpMat& KKT_K)
    {
        // do nothing, since factor has been done in 'getInertia' function
        return 0;
    }

    int solver::KKT_solve1(const Eigen::VectorXd& KKT_b, Eigen::VectorXd& x)
    {
        clock_.tic();

        x = KKT_b; // 重要! 别删这个

        spral_ssids_solve1(
            0, 
            x.data(), 
            factorInfo_.akeep, 
            factorInfo_.fkeep, 
            &factorInfo_.options, 
            &factorInfo_.inform
        );
        if(factorInfo_.inform.flag < 0) {
            printf("spral error as flag %d\n",factorInfo_.inform.flag);
            exit(1);
        }

        processRecorder_.n_solve += 1;
        processRecorder_.t_solve += clock_.toc(0);

        return factorInfo_.inform.flag;
    }

    // 求解两个线性方程组，x = [x1;x2]
    int solver::KKT_solve2(const Eigen::VectorXd& KKT_B, Eigen::VectorXd& x)
    {
        clock_.tic();

        x = KKT_B; // 重要! 别删这个

        spral_ssids_solve(
            0,
            2,
            x.data(),
            factorInfo_.n,
            factorInfo_.akeep,
            factorInfo_.fkeep,
            &factorInfo_.options,
            &factorInfo_.inform
        );
        if(factorInfo_.inform.flag < 0) {
            printf("spral error as flag %d\n",factorInfo_.inform.flag);
            exit(1);
        }

        processRecorder_.n_solve += 2;
        processRecorder_.t_solve += clock_.toc(0);

        return factorInfo_.inform.flag;
    }
};
# endif