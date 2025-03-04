// eigen lib
# include <Eigen/Core>
# include <Eigen/SparseCore>

# ifdef HOMO_OCP_USE_SPRAL
// spral lib
# include <spral.h>

// c lib
# include <stdlib.h>
# include <stdint.h>
# include <stdio.h>

struct factorInfo
{
    int n;
    Eigen::VectorXi ptr;
    Eigen::VectorXi row;

    void* akeep = nullptr;
    void* fkeep = nullptr;
    struct spral_ssids_options options;
    struct spral_ssids_inform inform;

    factorInfo()
    {
        spral_ssids_default_options(&options);
        options.print_level = -1;
        options.action = false;
    }
    
    ~factorInfo()
    {
        spral_ssids_free(&akeep, &fkeep);
    }

    factorInfo(const factorInfo&) = delete;
    factorInfo& operator = (const factorInfo&) = delete;
};

# elif defined(HOMO_OCP_USE_EIGEN)
// Eigen build in linear solver
# include <Eigen/SparseLU>

struct factorInfo
{
    Eigen::SparseLU<Eigen::SparseMatrix<double>> linSolver;

    factorInfo()
    {
        linSolver.isSymmetric(true);
    }
};

# endif
