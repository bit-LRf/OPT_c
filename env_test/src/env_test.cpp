#define EIGEN_USE_THREADS

# include <iostream>
# include <Eigen/Core>
# include <Eigen/Dense>
# include <Eigen/Sparse>
# include "spral.h"
# include <stdlib.h>
# include <stdint.h>
# include <stdio.h>
# include <math.h>
# include <vector>
# include <omp.h>
# include <chrono>

struct sol
{
    Eigen::ArrayXd x;
};


int main(void) {
    Eigen::initParallel();

    double tmp_1 = -0./0.;

    std::cout<<"hello cpp"<<std::isnan(tmp_1)<<std::endl;

    //=================for openMP====================
    omp_set_num_threads(16);
    #pragma omp parallel 
    {
    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    printf("Hello from thread %d out of %d threads\n", thread_id, num_threads);
    }

    //=================for eigen====================
    Eigen::Matrix2d mat2;
    mat2 << 5,6,
            7,8;
    std::cout<<mat2<<mat2.all()<<std::endl;

    Eigen::ArrayXd v(5);
    v << 1,2,3,4,5;
    std::vector<int> idx = {};
    idx.resize(4);
    idx = {2,3,4,1};
    std::cout<<"v(idx) = "<<v(idx).transpose()<<std::endl;

    v(idx) << 1,2,3,4;
    std::cout<<"v = "<<v.transpose()<<std::endl;

    std::cout << "(v > 0).all()   = " << (v > 0).all() << std::endl;

    sol m_sol;
    m_sol.x = v;

    Eigen::ArrayXXd v2(3,3);
    std::cout<<"v2:"<<v2<<std::endl;
    
    // 共享内存参考
    double data[] = {1,2,3,4,5};
    Eigen::Map<Eigen::VectorXd> tmp_df(data,5);
    Eigen::VectorXd tmp_df2 = tmp_df;
    std::cout<<tmp_df.transpose()<<std::endl;
    std::cout<<tmp_df2.transpose()<<std::endl;
    data[2] = 0;
    std::cout<<tmp_df.transpose()<<std::endl;
    std::cout<<tmp_df2.transpose()<<std::endl;

    // 稀疏矩阵测试
    Eigen::SparseMatrix<double,Eigen::ColMajor> spMat(5,5);
    Eigen::Map<Eigen::VectorXi> tmp_ptr1(const_cast<int*>(spMat.outerIndexPtr()), spMat.outerSize());
    std::cout<<spMat<<std::endl;
    std::cout<<tmp_ptr1.transpose()<<"\n"<<std::endl;

    spMat += Eigen::VectorXd::Constant(5,0).asDiagonal();
    Eigen::Map<Eigen::VectorXi> tmp_ptr2(const_cast<int*>(spMat.outerIndexPtr()), spMat.outerSize());
    std::cout<<spMat<<std::endl;
    std::cout<<tmp_ptr2.transpose()<<"\n"<<std::endl;

    spMat.coeffRef(2,3) += 0;
    spMat.makeCompressed();
    Eigen::Map<Eigen::VectorXi> tmp_ptr3(const_cast<int*>(spMat.outerIndexPtr()), spMat.outerSize());
    std::cout<<spMat<<std::endl;
    std::cout<<tmp_ptr3.transpose()<<"\n"<<std::endl;
    
    //=================for spral====================
    /* Derived types */
    void *akeep, *fkeep;
    struct spral_ssids_options options;
    struct spral_ssids_inform inform;

    /* Initialize derived types */
    akeep = NULL; fkeep = NULL; /* Important that these are NULL to start with */
    spral_ssids_default_options(&options);
    options.array_base = 1; /* Need to set to 1 if using Fortran 1-based indexing */

    /* Data for matrix:
    * ( 2  1         )
    * ( 1  4  1    1 )
    * (    1  3  2   )
    * (       2 -1   )
    * (    1       2 ) */
    bool posdef = false;
    int n = 5;
    int ptr[]   = { 1,        3,             6,         8,   9,  10 };
    int row[]    = { 1,   2,   2,   3,   5,   3,   4,    4,   5   };
    double val[] = { 2.0, 1.0, 4.0, 1.0, 1.0, 3.0, 2.0, -1.0, 2.0 };

    /* The right-hand side with solution (1.0, 2.0, 3.0, 4.0, 5.0) */
    double x[] = { 4.0, 17.0, 19.0, 2.0, 12.0 };

    auto start = std::chrono::high_resolution_clock::now();

    /* Perform analyse and factorise with data checking */
    bool check = true;
    spral_ssids_analyse_ptr32(check, n, NULL, ptr, row, NULL, &akeep, &options,
            &inform);
    if(inform.flag<0) {
        spral_ssids_free(&akeep, &fkeep);
        exit(1);
    }
    spral_ssids_factor(posdef, NULL, NULL, val, NULL, akeep, &fkeep, &options,
            &inform);
    if(inform.flag<0) {
        spral_ssids_free(&akeep, &fkeep);
        exit(1);
    }

    /* Solve */
    spral_ssids_solve1(0, x, akeep, fkeep, &options, &inform);
    if(inform.flag<0) {
        spral_ssids_free(&akeep, &fkeep);
        exit(1);
    }
    printf("The computed solution is:\n");
    for(int i=0; i<n; i++) printf(" %18.10e", x[i]);
    printf("\n");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;

    /* Determine and print the pivot order */
    int piv_order[5];
    spral_ssids_enquire_indef(akeep, fkeep, &options, &inform, piv_order, NULL);
    printf("Pivot order:");
    for(int i=0; i<n; i++) printf(" %5d", piv_order[i]);
    printf("\n");

    int cuda_error = spral_ssids_free(&akeep, &fkeep);
    if(cuda_error!=0) exit(1);

    std::cout<<"env_test pass!"<<std::endl;

    return 0;
}
