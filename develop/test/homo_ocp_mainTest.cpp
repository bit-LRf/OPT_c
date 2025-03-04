# ifndef EIGEN_USE_BLAS
# define EIGEN_USE_BLAS
# endif

# include "homo_ocp.h"
# include "testProb_yourSuffix.h"

void test1()
{
    homo_ocp::prob m_prob;
    HOMO_OCP_PROB(m_prob,testProb)
    if (m_prob.n_x != nz_h[0] || m_prob.n_x != nz_A[1] || m_prob.n_eq != nz_A[0])
    {
        throw std::runtime_error("问题维度不一致，请检查提供问题及其梯度信息的文件");
    }

    m_prob.p = Eigen::VectorXd::Constant(m_prob.n_p,0.01);
    m_prob.lbx = Eigen::VectorXd::Constant(m_prob.n_x,0);
    m_prob.ubx = Eigen::VectorXd::Constant(m_prob.n_x,2.1);

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(m_prob.n_x,1);
    homo_ocp::solver m_solver(m_prob,9999,12,1);
    
    // 修改参数
    m_solver.basicParam_.useHomotopy = true;
    // m_solver.barrier_.updateMode_ = homo_ocp::barrier::updateMode::Monotone; 

    // 求解
    auto sol = m_solver.solve(x0);
    sol = m_solver.solve(x0);
    std::cout<<sol.x.transpose()<<std::endl;
};

void test2()
{
    homo_ocp::SpMat H(2,2);
    H.coeffRef(0,0) = 1;
    H.coeffRef(1,0) = -1;
    H.coeffRef(0,1) = -1;
    H.coeffRef(1,1) = 2;

    Eigen::VectorXd c(2);
    c << -2,-6;

    homo_ocp::SpMat A_eq(0,0);
    homo_ocp::SpMat A_ineq(3,2);
    A_ineq.coeffRef(0,0) = -1;
    A_ineq.coeffRef(1,0) = 1;
    A_ineq.coeffRef(2,0) = -2;
    A_ineq.coeffRef(0,1) = -1;
    A_ineq.coeffRef(1,1) = -2;
    A_ineq.coeffRef(2,1) = -1;

    Eigen::VectorXd b_eq(0);
    Eigen::VectorXd b_ineq(3);
    b_ineq << -2,-2,-3;

    Eigen::VectorXd lbx = Eigen::VectorXd::Constant(2,-9999);
    Eigen::VectorXd ubx = Eigen::VectorXd::Constant(2,9999);

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2,1);
    homo_ocp::solver m_solver(H,c,A_eq,b_eq,A_ineq,b_ineq,lbx,ubx,9999,12,1);
    
    // 修改参数
    m_solver.basicParam_.useHomotopy = true;
    // m_solver.barrier_.updateMode_ = homo_ocp::barrier::updateMode::Monotone; 

    // 求解
    auto sol = m_solver.solve(x0);
    sol = m_solver.solve(x0);
    std::cout<<sol.x.transpose()<<std::endl;
}

void test3()
{
    int n = 102400;

    omp_set_num_threads(8);

    homo_ocp::clock clock_;

    double lbx = 0;
    double ubx = 1;

    auto r = Eigen::VectorXd::Random(n);
    Eigen::VectorXd a(n);
    clock_.tic();
    for (size_t i = 0; i < n; i++)
    {
        a(i) += std::max(std::min(r(i),ubx),lbx);
    }
    clock_.toc();
    std::cout<<"single\n"<<std::endl;
    
    Eigen::VectorXd b(n);
    clock_.tic();
    # pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        b(i) += std::max(std::min(r(i),ubx),lbx);
    }
    clock_.toc();
    std::cout<<"parallel\n"<<std::endl;
}

int main(int argc, char const *argv[])
{
    test1();

    test2();

    test3();
    
    return 0;
}