# include "homo_ocp.h"

namespace homo_ocp
{
    void solver::showThread()
    {
        printf("This is thread %d out of %d threads, running at %dth CPU core\n", omp_get_thread_num() + 1, omp_get_num_threads(),sched_getcpu());
    };

    void solver::showSpMat(SpMat& m_SpMat)
    {
        m_SpMat.makeCompressed();
        
        // 提取CSC格式
        std::cout<<"nonZeros = "<<m_SpMat.nonZeros()<<std::endl;

        std::cout<<"m_SpMat_ptr: "<<std::endl;
        for (size_t i = 0; i < m_SpMat.outerSize(); i++)
        {
            std::cout<<m_SpMat.outerIndexPtr()[i]<<", ";
        }
        std::cout<<std::endl;

        std::cout<<"m_SpMat_row: "<<std::endl;
        for (size_t i = 0; i < m_SpMat.nonZeros(); i++)
        {
            std::cout<<m_SpMat.innerIndexPtr()[i]<<", ";
        }
        std::cout<<std::endl;

        std::cout<<"m_SpMat_val: "<<std::endl;
        for (size_t i = 0; i < m_SpMat.nonZeros(); i++)
        {
            std::cout<<m_SpMat.valuePtr()[i]<<", ";
        }
        std::cout<<std::endl;
    }

    void solver::showProcess()
    {
        printf("KKT error at solutions:%.16f\n\n",sol_KKT_error);

        printf("tol iterations .................................................%d\n\n",iter);

        printf("number of f evaluations ........................................%d\n",processRecorder_.n_f);
        printf("number of jacobian of f evaluations ............................%d\n",processRecorder_.n_df);
        printf("number of hessian of f evaluations .............................%d\n",processRecorder_.n_h);
        printf("number of equations evaluations ................................%d\n",processRecorder_.n_eq);
        printf("number of jacobian of equations evaluations ....................%d\n",processRecorder_.n_deq);
        printf("number of KKT system factorizations ............................%d\n",processRecorder_.n_factor);
        printf("number of solving times of KKT system ..........................%d\n\n",processRecorder_.n_solve);
        
        printf("tol cpu time ...................................................%f ms\n\n",cpuTime);

        printf("time for f evaluations .........................................%f ms\n",processRecorder_.t_f);
        printf("time for jacobian of f evaluations .............................%f ms\n",processRecorder_.t_df);
        printf("time for hessian of f evaluations ..............................%f ms\n",processRecorder_.t_h);
        printf("time for equations evaluations .................................%f ms\n",processRecorder_.t_eq);
        printf("time for jacobian of equations evaluations .....................%f ms\n",processRecorder_.t_deq);
        printf("time for KKT system factorizations .............................%f ms\n",processRecorder_.t_factor);
        printf("time for solving KKT system ...........................,,.......%f ms\n\n",processRecorder_.t_solve);
    }

    const prob *solver::getProb()
    {
        return &prob_;
    }

    const sol *solver::getSolution()
    {
        return &sol_;
    }

    int solver::getIteration()
    {
        return iter;
    }

    double solver::getCpuTime()
    {
        return cpuTime;
    }

    int solver::getHomotopyConvergeIteration()
    {
        return homotopyConvergeIter;
    }

    const processRecorder *solver::getProcessInfo()
    {
        return &processRecorder_;
    }

    int solver::getExitFlag()
    {
        return exit_flag;
    }

    const wsInfo *solver::getWsInfo()
    {
        return &wsInfo_;
    }
} // namespace homo_ocp
