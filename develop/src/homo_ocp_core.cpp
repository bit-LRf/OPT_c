# include "homo_ocp.h"

namespace homo_ocp
{
    sol solver::solve(const Eigen::VectorXd& x0)
    {
        cpuTime = 0;
        clock tmp_clock;
        tmp_clock.tic();

        if (basicParam_.printLevel >= 2)
        {
            printf("\nstart solving ...\n");
        }
        
        if (!refresh_flag)
        {
            refresh();
        }
        
        // 赋初值
        bool isLambdaInit = false;
        Eigen::VectorXd p = prob_.p;
        if (wsInfo_.isInit == 0 || !basicParam_.useHomotopy)
        {
            if (basicParam_.printLevel >= 2)
            {
                printf("cold start\n");
            }

            sol_.x = x0;
            sol_.lambda = Eigen::VectorXd::Constant(prob_.n_eq,basicParam_.accept_tol);
            sol_.s_lbx = Eigen::VectorXd::Zero(prob_.n_x);
            sol_.s_ubx = Eigen::VectorXd::Zero(prob_.n_x);
            sol_.s_lbx(idx_lbx).array() = 1;
            sol_.s_ubx(idx_ubx).array() = 1;

            if (isQPprob)
            {
                sol_.x.resize(prob_.n_x);
                sol_.x.topRows(x0.size()) = x0;
                sol_.x.bottomRows(prob_.n_x - x0.size()) = QP_A.bottomLeftCorner(prob_.n_x - x0.size(),x0.size())*x0 - QP_b.bottomRows(prob_.n_x - x0.size());
            }
            
            // 内点法需要保证起点满足不等式约束
            # pragma omp parallel for
            for (size_t i = 0; i < prob_.n_x; i++)
            {
                if (std::abs(prob_.lbx(i)) < inf_ && std::abs(prob_.ubx(i)) < inf_)
                {
                    // 双边界
                    sol_.x(i) = std::min(
                        std::max(sol_.x(i),prob_.lbx(i) + basicParam_.kappa_2*(prob_.ubx(i) - prob_.lbx(i))),
                        prob_.ubx(i) - basicParam_.kappa_2*(prob_.ubx(i) - prob_.lbx(i)));

                }
                else if (std::abs(prob_.lbx(i)) < inf_)
                {
                    // 仅下界
                    sol_.x(i) = std::max(sol_.x(i),prob_.lbx(i) + basicParam_.kappa_1*std::max(1.,std::abs(prob_.lbx(i))));
                }
                else if (std::abs(prob_.ubx(i)) < inf_)
                {
                    // 仅上界
                    sol_.x(i) = std::min(sol_.x(i),prob_.ubx(i) - basicParam_.kappa_1*std::max(1.,std::abs(prob_.ubx(i))));
                }
            }

            homotopy_.lota = 1;
            if (basicParam_.useHomotopy == false)
            {
                homotopy_.isDone = true;
            }
        }
        else
        {
            if (basicParam_.printLevel >= 2)
            {
                printf("warm start\n");
            }

            sol_ = wsInfo_.sol_;
            
            isLambdaInit = true;

            if ((prob_.p.array() == wsInfo_.p.array()).all())
            {
                if (basicParam_.printLevel >= 2)
                {
                    printf("homotopy path following process done since parameters did not change\n");
                }
                homotopy_.lota = 1;
                homotopy_.isDone = true;
            }
            else
            {
                homotopy_.lota = homotopy_.diff_lota_min;
                p = prob_.p*homotopy_.lota + wsInfo_.p*(1 - homotopy_.lota);
            }
        }
        int homotopyUpdateIter = 0;

        double KKT_error;
        iter = 0;
        homotopyConvergeIter = 0;
        while (1)
        {
            SpMat KKT_K(prob_.n_x + prob_.n_eq,prob_.n_x + prob_.n_eq); // 这个矩阵必须在每次使用前重新定义
            KKT_K.reserve(mapNNZ_KKT_K);

            // 
            v_lbx_active = sol_.x(idx_lbx) - prob_.lbx(idx_lbx);
            v_ubx_active = -sol_.x(idx_ubx) + prob_.ubx(idx_ubx);

            s_lbx_active = sol_.s_lbx(idx_lbx);
            s_ubx_active = sol_.s_ubx(idx_ubx);

            val_df.setZero();
            get_val_df(sol_.x,p,val_df,1);

            // 构造牛顿步的左边
            get_val_deq(sol_.x,p,KKT_K,-1);
            A = -KKT_K.bottomLeftCorner(prob_.n_eq,prob_.n_x);

            if (!isLambdaInit && prob_.n_eq != 0)
            {
                isLambdaInit = true;

                KKT_b.setZero();
                KKT_b(Eigen::seq(0,prob_.n_x - 1)) = -(val_df - sol_.s_lbx + sol_.s_ubx);

                tmp_K = KKT_K + fullZeroK;
                tmp_K += diag_Ix.asDiagonal();
                tmp_K += 0*diag_Ieq.asDiagonal();
                if (KKT_factor(tmp_K) >= 0)
                {
                    KKT_solve1(KKT_b,tmp_sol);
                    sol_.lambda = tmp_sol(Eigen::lastN(prob_.n_eq));
                }

                if (sol_.lambda.lpNorm<Eigen::Infinity>() >= basicParam_.lambda_max || sol_.lambda.array().isNaN().any())
                {
                    sol_.lambda = Eigen::VectorXd::Constant(prob_.n_eq,basicParam_.accept_tol);
                }
            }

            // 计算kkt系统残差
            r_x = val_df - sol_.s_lbx + sol_.s_ubx;
            if (prob_.n_eq != 0)
            {
                r_x -= (sol_.lambda.transpose()*A).transpose();
            }
            r_lambda.setZero();
            get_val_eq(sol_.x,p,r_lambda,1);
            r_lbxs = v_lbx_active.array()*s_lbx_active.array();
            r_ubxs = v_ubx_active.array()*s_ubx_active.array();

            KKT_error = std::max({
                r_x.lpNorm<Eigen::Infinity>(), 
                r_lambda.lpNorm<Eigen::Infinity>(),
                r_lbxs.lpNorm<Eigen::Infinity>(),
                r_ubxs.lpNorm<Eigen::Infinity>()});

            if (basicParam_.printLevel >= 2)
            {
                printf("KKT_error = %.16f\n",KKT_error);
            }

            if (homotopy_.isDone == false)
            {
                double homoError = std::max({
                r_x.lpNorm<Eigen::Infinity>(), 
                r_lambda.lpNorm<Eigen::Infinity>(),
                (r_lbxs.array() - barrier_.mu).matrix().lpNorm<Eigen::Infinity>(),
                (r_ubxs.array() - barrier_.mu).matrix().lpNorm<Eigen::Infinity>()});
                if (homoError <= homotopy_.epsilon*basicParam_.accept_tol && watchDog_.iter == 0)
                {
                    if (homotopy_.lota < 1 && iter > 0 && homotopyUpdateIter >= 1)
                    {
                        double lota_old = homotopy_.lota;
                        double diff_lota = std::max(std::pow(homotopy_.diff_lota_max,1 + std::log10(homotopyUpdateIter)),homotopy_.diff_lota_min);
                        homotopy_.lota += diff_lota;
                        if (homotopy_.lota >= 1)
                        {
                            homotopy_.lota = 1;
                        }
                        p = prob_.p*homotopy_.lota + wsInfo_.p*(1 - homotopy_.lota);

                        filter_.refresh();

                        if (basicParam_.printLevel >= 2)
                        {
                            printf("lota = %f -> %f, last homotopy iteration: %d\n",lota_old,homotopy_.lota,homotopyUpdateIter);
                        }
                        
                        homotopyUpdateIter = 0;

                        continue;
                    }
                    else if (homotopy_.lota == 1)
                    {
                        homotopy_.isDone = true;
                        homotopyConvergeIter = iter;

                        wsInfo_.sol_ = sol_;
                        wsInfo_.p = prob_.p;
                        wsInfo_.isInit = true;

                        if (basicParam_.printLevel >= 2)
                        {
                            printf("homotopy path following process done, toltal iteration: %d\n",homotopyConvergeIter);
                        }
                    }
                }

                if (iter >= homotopy_.iter_max)
                {
                    homotopy_.isDone = true;
                    homotopyConvergeIter = iter;

                    wsInfo_.sol_ = sol_;
                    wsInfo_.p = p;
                    wsInfo_.isInit = true;

                    if (basicParam_.printLevel >= 2)
                    {
                        std::cout<<"homotopy path following process exit because of the maximum iteration, ";
                        std::cout<<"toltal iteration: "<<homotopyConvergeIter<<std::endl;
                    }
                }
                
                homotopyUpdateIter += 1;
            }
            
            // 停机条件
            if (KKT_error <= basicParam_.accept_tol && homotopy_.isDone == true)
            {
                exit_flag = 1;
                break;
            }
            if (iter >= basicParam_.iter_max && homotopy_.isDone == true)
            {
                exit_flag = 0;
                break;
            }
            iter = iter + 1;

            if (basicParam_.printLevel >= 2)
            {
                printf("\n===========================iteration %d=============================\n",iter);
                if (watchDog_.iter > 0)
                {
                    printf("watchDog activating: \n");
                    printf("   iter = %d\n",watchDog_.iter);
                    printf("   merit_sigma = %.8f\n",watchDog_.merit_sigma);
                }
            }

            // 构造K矩阵
            get_val_h(sol_.x,sol_.lambda,p,KKT_K,1);
            H = KKT_K.topLeftCorner(prob_.n_x,prob_.n_x);

            d.setZero();
            d(idx_lbx) = s_lbx_active.array()/v_lbx_active.array();
            d(idx_ubx) = d(idx_ubx).array() + s_ubx_active.array()/v_ubx_active.array();
            d0.setZero();
            d0(Eigen::seq(0,prob_.n_x - 1)) = d;
            KKT_K += d0.asDiagonal();
            KKT_K += 0*diag_Ieq.asDiagonal();

            // 分解K矩阵并修正惯性指数
            int IC_count_w = 0;
            int IC_count_a = 0;
            if (ICor_.wCorFlag == 1 && ICor_.w_count >= 1)
            {
                KKT_K += ICor_.w*diag_Ix.asDiagonal();

                ICor_.w = ICor_.w*ICor_.w_ac;
                IC_count_w += 1;
            }
            if (ICor_.aCorFlag == 1 && ICor_.a_count >= 1)
            {
                KKT_K -= ICor_.a*diag_Ieq.asDiagonal();

                ICor_.a = ICor_.a*ICor_.a_ac;
                IC_count_a += 1;
            }
            while (1)
            {
                std::pair<bool,int> IC = getInertia(KKT_K);

                bool tmp_flag = true;
                if (IC.first)
                {
                    KKT_K += ICor_.a*diag_Ix.asDiagonal();
                    KKT_K -= ICor_.a*diag_Ieq.asDiagonal();

                    ICor_.a = ICor_.a*ICor_.a_ac;
                    IC_count_a += 1;
                    tmp_flag = 0;
                }
                if (IC.second > prob_.n_eq)
                {
                    KKT_K += ICor_.w*diag_Ix.asDiagonal();

                    ICor_.w = ICor_.w*ICor_.w_ac;
                    IC_count_w += 1;
                    tmp_flag = 0;
                }
                
                if (ICor_.w >= ICor_.w_max)
                {
                    ICor_.w = ICor_.w_max;

                    break;
                }
                if (ICor_.a >= ICor_.a_max)
                {
                    ICor_.a = ICor_.a_max;

                    break;
                }
                
                if (tmp_flag == 1)
                {
                    break;
                }
            }

            KKT_factor_final(KKT_K);

            if (ICor_.wCorFlag == 0)
            {
                if (IC_count_w > 0)
                {
                    ICor_.w_count += 1;

                    if (ICor_.w_count >= ICor_.switchIter)
                    {
                        ICor_.w_count = 0;
                        ICor_.wCorFlag = 1;
                    }
                }
                else
                {
                    ICor_.w_count = 0;
                }
            }
            else if (ICor_.wCorFlag == 1)
            {
                if (ICor_.w_count == 0 && IC_count_w == 0)
                {
                    ICor_.w_count_2 += 1;

                    if (ICor_.w_count_2 >= ICor_.switchIter)
                    {
                        ICor_.w_count = 0;
                        ICor_.w_count_2 = 0;
                        ICor_.wCorFlag = 0;
                    }                    
                }
                else
                {
                    ICor_.w_count_2 = 0;
                    ICor_.w_count += 1;

                    if (ICor_.w_count >= ICor_.porbeIter)
                    {
                        ICor_.w_count = 0;
                    }
                }
            }

            if (ICor_.aCorFlag == 0)
            {
                if (IC_count_a > 0)
                {
                    ICor_.a_count += 1;

                    if (ICor_.a_count >= ICor_.switchIter)
                    {
                        ICor_.a_count = 0;
                        ICor_.aCorFlag = 1;
                    }
                }
                else
                {
                    ICor_.a_count = 0;
                }
            }
            else if (ICor_.aCorFlag == 1)
            {
                if (ICor_.a_count == 0 && IC_count_a == 0)
                {
                    ICor_.a_count_2 += 1;

                    if (ICor_.a_count_2 >= ICor_.switchIter)
                    {
                        ICor_.a_count = 0;
                        ICor_.a_count_2 = 0;
                        ICor_.aCorFlag = 0;
                    }                    
                }
                else
                {
                    ICor_.a_count_2 = 0;
                    ICor_.a_count += 1;

                    if (ICor_.a_count >= ICor_.porbeIter)
                    {
                        ICor_.a_count = 0;
                    }
                }
            }
            
            if (basicParam_.printLevel >= 2)
            {
                printf("hessian correction times: %d, deq correction times: %d\n",IC_count_w,IC_count_a);
            }

            if (IC_count_a == 0)
            {
                ICor_.a = std::max(ICor_.a_min,ICor_.a/ICor_.a_dc);
            }
            else
            {
                ICor_.a = std::max(ICor_.a_min,ICor_.a/ICor_.a_dc/ICor_.a_ac);
            }
            if (IC_count_w == 0)
            {
                ICor_.w = std::max(ICor_.w_min,ICor_.w/ICor_.w_dc);
            }
            else
            {
                ICor_.w = std::max(ICor_.w_min,ICor_.w/ICor_.w_dc/ICor_.w_ac);
            }
            
            // 更新罚参数
            double dual_measurement = (r_lbxs.array().sum() + r_ubxs.array().sum())/(n_lbx + n_ubx);
            if (n_lbx + n_ubx == 0)
            {
                dual_measurement = 0;
            }
            
            double mu_old = barrier_.mu;
            bool solve_flag = 0;
            if (homotopy_.isDone == false)
            {
                // 当处于参数同伦路径跟踪阶段时不更新罚参数
            }
            else if (barrier_.updateMode_ == barrier::updateMode::Monotone || barrier_.isProtected)
            {
                if (KKT_error <= basicParam_.monotone.epsilon*barrier_.mu)
                {
                    barrier_.mu = std::min(basicParam_.monotone.k*barrier_.mu,
                        std::pow(barrier_.mu,basicParam_.monotone.t));

                    barrier_.mu = std::max(basicParam_.accept_tol/basicParam_.monotone.epsilon,
                        barrier_.mu);

                    if (basicParam_.printLevel >= 2)
                    {
                        printf("Monotone mode/Protected barrier parameter update\n");
                    }
                }
            }
            else if (barrier_.updateMode_ == barrier::updateMode::LOQO_rule)
            {
                double xi = std::min(r_lbxs.array().minCoeff(),r_ubxs.array().minCoeff())/dual_measurement;
                double sigma = basicParam_.loqo.alpha*std::pow(std::min(basicParam_.loqo.k*(1 - xi)/xi,basicParam_.loqo.t),3);

                barrier_.mu = std::max(sigma*dual_measurement,basicParam_.loqo.mu_min);
            }
            else if (barrier_.updateMode_ == barrier::updateMode::QualityFunction)
            {
                KKT_B.setZero();

                r_lbxs_qf_0 = r_lbxs.array() - 0*dual_measurement;
                r_ubxs_qf_0 = r_ubxs.array() - 0*dual_measurement;
                R_lbx_qf_0.setZero();
                R_ubx_qf_0.setZero();
                R_lbx_qf_0(idx_lbx) = r_lbxs_qf_0.array()/v_lbx_active.array();
                R_ubx_qf_0(idx_ubx) = r_ubxs_qf_0.array()/v_ubx_active.array();
                KKT_B(Eigen::seq(0,prob_.n_x - 1)) = -r_x - R_lbx_qf_0 + R_ubx_qf_0;
                KKT_B(Eigen::seq(prob_.n_x,prob_.n_x + prob_.n_eq - 1)) = r_lambda;

                int n = prob_.n_x + prob_.n_eq;
                r_lbxs_qf_1 = r_lbxs.array() - 1*dual_measurement;
                r_ubxs_qf_1 = r_ubxs.array() - 1*dual_measurement;
                R_lbx_qf_1.setZero();
                R_ubx_qf_1.setZero();
                R_lbx_qf_1(idx_lbx) = r_lbxs_qf_1.array()/v_lbx_active.array();
                R_ubx_qf_1(idx_ubx) = r_ubxs_qf_1.array()/v_ubx_active.array();
                KKT_B(Eigen::seq(n,n + prob_.n_x - 1)) = -r_x - R_lbx_qf_1 + R_ubx_qf_1;
                KKT_B(Eigen::seq(n + prob_.n_x,n + prob_.n_x + prob_.n_eq - 1)) = r_lambda;

                diff_qf.setZero();
                KKT_solve2(KKT_B,diff_qf);
                
                diff_x_qf_0 = diff_qf(Eigen::seq(0,prob_.n_x - 1));
                diff_lambda_qf_0 = diff_qf(Eigen::seq(prob_.n_x,prob_.n_x + prob_.n_eq - 1));
                diff_s_lbx_qf_0.setZero();
                diff_s_lbx_qf_0(idx_lbx) = -R_lbx_qf_0(idx_lbx)
                    - (s_lbx_active.array()*diff_x_qf_0(idx_lbx).array()/v_lbx_active.array()).matrix();
                diff_s_ubx_qf_0.setZero();
                diff_s_ubx_qf_0(idx_ubx) = -R_ubx_qf_0(idx_ubx)
                    + (s_ubx_active.array()*diff_x_qf_0(idx_ubx).array()/v_ubx_active.array()).matrix();

                int n_xpeq = prob_.n_x + prob_.n_eq;
                diff_x_qf_1 = diff_qf(Eigen::seq(n_xpeq,n_xpeq + prob_.n_x - 1));
                diff_lambda_qf_1 = diff_qf(Eigen::seq(n_xpeq + prob_.n_x,n_xpeq + prob_.n_x + prob_.n_eq - 1));
                diff_s_lbx_qf_1.setZero();
                diff_s_lbx_qf_1(idx_lbx) = -R_lbx_qf_1(idx_lbx)
                    - (s_lbx_active.array()*diff_x_qf_1(idx_lbx).array()/v_lbx_active.array()).matrix();
                diff_s_ubx_qf_1.setZero();
                diff_s_ubx_qf_1(idx_ubx) = -R_ubx_qf_1(idx_ubx)
                    + (s_ubx_active.array()*diff_x_qf_1(idx_ubx).array()/v_ubx_active.array()).matrix();

                P1_qf_0 = H.selfadjointView<Eigen::Lower>()*diff_x_qf_0;
                P2_qf_0 = A*diff_x_qf_0;
                P3_qf_0 = (diff_lambda_qf_0.transpose()*A).transpose();

                P1_qf_1 = H.selfadjointView<Eigen::Lower>()*diff_x_qf_1;
                P2_qf_1 = A*diff_x_qf_1;
                P3_qf_1 = (diff_lambda_qf_1.transpose()*A).transpose();

                // 确定搜索区间
                double sigma_mid = basicParam_.quality.mu_min/dual_measurement;
                if (std::isnan(sigma_mid) || std::isinf(sigma_mid))
                {
                    sigma_mid = 0;
                }

                double sigma_min;
                double sigma_max;
                sigma_min = std::max(basicParam_.quality.sigma_min,sigma_mid);
                if (sigma_min < basicParam_.quality.sigma_max)
                {
                    sigma_max = basicParam_.quality.sigma_max;
                }
                else
                {
                    sigma_max = std::pow(10,basicParam_.quality.block)*sigma_min;
                }
                
                Eigen::VectorXd sigmaVector(basicParam_.quality.block + 1);
                sigmaVector(0) = sigma_min;
                sigmaVector(sigmaVector.size() - 1) = sigma_max;

                double tmp_p = (std::log10(sigma_max) - log10(sigma_min))/basicParam_.quality.block;
                for (size_t i = 1; i < sigmaVector.size() - 1; i++)
                {
                    sigmaVector(i) = sigma_min*std::pow(10,i*tmp_p);
                }

                Eigen::VectorXd qfVector(basicParam_.quality.block + 1);
                for (size_t i = 0; i < qfVector.size(); i++)
                {
                    tmp_diff_x = sigmaVector(i)*diff_x_qf_1 + (1 - sigmaVector(i))*diff_x_qf_0;
                    tmp_diff_s_lbx = sigmaVector(i)*diff_s_lbx_qf_1+ (1 - sigmaVector(i))*diff_s_lbx_qf_0;
                    tmp_diff_s_ubx = sigmaVector(i)*diff_s_ubx_qf_1+ (1 - sigmaVector(i))*diff_s_ubx_qf_0;

                    tmp_P1 = sigmaVector(i)*P1_qf_1 + (1 - sigmaVector(i))*P1_qf_0;
                    tmp_P2 = sigmaVector(i)*P2_qf_1 + (1 - sigmaVector(i))*P2_qf_0;
                    tmp_P3 = sigmaVector(i)*P3_qf_1 + (1 - sigmaVector(i))*P3_qf_0;

                    std::pair<double,double> tmp_alpha = get_stepLength(tmp_diff_x,tmp_diff_s_lbx,tmp_diff_s_ubx,1);
                    qfVector(i) = quality_function(tmp_diff_x,tmp_diff_s_lbx,tmp_diff_s_ubx,tmp_P1,tmp_P2,tmp_P3,tmp_alpha);
                }
                
                double sigma_qf_a;
                double sigma_qf_b;
                Eigen::Index tmp_ptr;
                qfVector.minCoeff(&tmp_ptr);
                if (tmp_ptr == 0)
                {
                    sigma_qf_a = sigmaVector(tmp_ptr);
                    sigma_qf_b = sigmaVector(tmp_ptr + 1);
                }
                else if (tmp_ptr == qfVector.size() - 1)
                {
                    sigma_qf_a = sigmaVector(tmp_ptr - 1);
                    sigma_qf_b = sigmaVector(tmp_ptr);
                }
                else
                {
                    if (qfVector(tmp_ptr - 1) <= qfVector(tmp_ptr + 1))
                    {
                        sigma_qf_a = sigmaVector(tmp_ptr - 1);
                        sigma_qf_b = sigmaVector(tmp_ptr);
                    }
                    else
                    {
                        sigma_qf_a = sigmaVector(tmp_ptr);
                        sigma_qf_b = sigmaVector(tmp_ptr + 1);
                    }
                }

                double tmp_t = (std::sqrt(5) - 1)/2;
                double tmp_h = sigma_qf_b - sigma_qf_a;
                double qf_p;
                double qf_q;
                double sigma_qf_p;
                double sigma_qf_q;
                std::pair<double,double> alpha_qf_p;
                std::pair<double,double> alpha_qf_q;

                sigma_qf_p = sigma_qf_a + (1 - tmp_t)*tmp_h;

                diff_x_qf_p = sigma_qf_p*diff_x_qf_1 + (1 - sigma_qf_p)*diff_x_qf_0;
                diff_lambda_qf_p = sigma_qf_p*diff_lambda_qf_1 + (1 - sigma_qf_p)*diff_lambda_qf_0;
                diff_s_lbx_qf_p = sigma_qf_p*diff_s_lbx_qf_1 + (1 - sigma_qf_p)*diff_s_lbx_qf_0;
                diff_s_ubx_qf_p = sigma_qf_p*diff_s_ubx_qf_1 + (1 - sigma_qf_p)*diff_s_ubx_qf_0;

                P1_qf_p = sigma_qf_p*P1_qf_1 + (1 - sigma_qf_p)*P1_qf_0;
                P2_qf_p = sigma_qf_p*P2_qf_1 + (1 - sigma_qf_p)*P2_qf_0;
                P3_qf_p = sigma_qf_p*P3_qf_1 + (1 - sigma_qf_p)*P3_qf_0;

                alpha_qf_p = get_stepLength(diff_x_qf_p,diff_s_lbx_qf_p,diff_s_ubx_qf_p,1);
                qf_p = quality_function(diff_x_qf_p,diff_s_lbx_qf_p,diff_s_ubx_qf_p,P1_qf_p,P2_qf_p,P3_qf_p,alpha_qf_p);

                sigma_qf_q = sigma_qf_a + tmp_t*tmp_h;

                diff_x_qf_q = sigma_qf_q*diff_x_qf_1 + (1 - sigma_qf_q)*diff_x_qf_0;
                diff_lambda_qf_q = sigma_qf_q*diff_lambda_qf_1 + (1 - sigma_qf_q)*diff_lambda_qf_0;
                diff_s_lbx_qf_q = sigma_qf_q*diff_s_lbx_qf_1 + (1 - sigma_qf_q)*diff_s_lbx_qf_0;
                diff_s_ubx_qf_q = sigma_qf_q*diff_s_ubx_qf_1 + (1 - sigma_qf_q)*diff_s_ubx_qf_0;

                P1_qf_q = sigma_qf_q*P1_qf_1 + (1 - sigma_qf_q)*P1_qf_0;
                P2_qf_q = sigma_qf_q*P2_qf_1 + (1 - sigma_qf_q)*P2_qf_0;
                P3_qf_q = sigma_qf_q*P3_qf_1 + (1 - sigma_qf_q)*P3_qf_0;

                alpha_qf_q = get_stepLength(diff_x_qf_q,diff_s_lbx_qf_q,diff_s_ubx_qf_q,1);
                qf_q = quality_function(diff_x_qf_q,diff_s_lbx_qf_q,diff_s_ubx_qf_q,P1_qf_q,P2_qf_q,P3_qf_q,alpha_qf_q);

                // 黄金分割
                int qf_iter = 0;
                while (1)
                {
                    qf_iter += 1;

                    if (qf_p <= qf_q)
                    {
                        // b = q
                        sigma_qf_b = sigma_qf_q;

                        // q = p
                        sigma_qf_q = sigma_qf_p;

                        diff_x_qf_q = diff_x_qf_p;
                        diff_lambda_qf_q = diff_lambda_qf_p;
                        diff_s_lbx_qf_q = diff_s_lbx_qf_p;
                        diff_s_ubx_qf_q = diff_s_ubx_qf_p;

                        alpha_qf_q = alpha_qf_p;
                        qf_q = qf_p;

                        // 
                        tmp_h = sigma_qf_b - sigma_qf_a;

                        // p = a + (1 - t)*h
                        sigma_qf_p = sigma_qf_a + (1 - tmp_t)*tmp_h;
                        
                        diff_x_qf_p = sigma_qf_p*diff_x_qf_1 + (1 - sigma_qf_p)*diff_x_qf_0;
                        diff_lambda_qf_p = sigma_qf_p*diff_lambda_qf_1 + (1 - sigma_qf_p)*diff_lambda_qf_0;
                        diff_s_lbx_qf_p = sigma_qf_p*diff_s_lbx_qf_1 + (1 - sigma_qf_p)*diff_s_lbx_qf_0;
                        diff_s_ubx_qf_p = sigma_qf_p*diff_s_ubx_qf_1 + (1 - sigma_qf_p)*diff_s_ubx_qf_0;

                        P1_qf_p = sigma_qf_p*P1_qf_1 + (1 - sigma_qf_p)*P1_qf_0;
                        P2_qf_p = sigma_qf_p*P2_qf_1 + (1 - sigma_qf_p)*P2_qf_0;
                        P3_qf_p = sigma_qf_p*P3_qf_1 + (1 - sigma_qf_p)*P3_qf_0;

                        alpha_qf_p = get_stepLength(diff_x_qf_p,diff_s_lbx_qf_p,diff_s_ubx_qf_p,1);
                        qf_p = quality_function(diff_x_qf_p,diff_s_lbx_qf_p,diff_s_ubx_qf_p,P1_qf_p,P2_qf_p,P3_qf_p,alpha_qf_p);
                    }
                    else
                    {
                        // a = p
                        sigma_qf_a = sigma_qf_p;

                        // p = q
                        sigma_qf_p = sigma_qf_q;

                        diff_x_qf_p = diff_x_qf_q;
                        diff_lambda_qf_p = diff_lambda_qf_q;
                        diff_s_lbx_qf_p = diff_s_lbx_qf_q;
                        diff_s_ubx_qf_p = diff_s_ubx_qf_q;

                        P1_qf_p = P1_qf_q;
                        P2_qf_p = P2_qf_q;
                        P3_qf_p = P3_qf_q;

                        alpha_qf_p = alpha_qf_q;
                        qf_p = qf_q;

                        // 
                        tmp_h = sigma_qf_b - sigma_qf_a;

                        // q = a + t*h
                        sigma_qf_q = sigma_qf_a + tmp_t*tmp_h;

                        diff_x_qf_q = sigma_qf_q*diff_x_qf_1 + (1 - sigma_qf_q)*diff_x_qf_0;
                        diff_lambda_qf_q = sigma_qf_q*diff_lambda_qf_1 + (1 - sigma_qf_q)*diff_lambda_qf_0;
                        diff_s_lbx_qf_q = sigma_qf_q*diff_s_lbx_qf_1 + (1 - sigma_qf_q)*diff_s_lbx_qf_0;
                        diff_s_ubx_qf_q = sigma_qf_q*diff_s_ubx_qf_1 + (1 - sigma_qf_q)*diff_s_ubx_qf_0;

                        P1_qf_q = sigma_qf_q*P1_qf_1 + (1 - sigma_qf_q)*P1_qf_0;
                        P2_qf_q = sigma_qf_q*P2_qf_1 + (1 - sigma_qf_q)*P2_qf_0;
                        P3_qf_q = sigma_qf_q*P3_qf_1 + (1 - sigma_qf_q)*P3_qf_0;

                        alpha_qf_q = get_stepLength(diff_x_qf_q,diff_s_lbx_qf_q,diff_s_ubx_qf_q,1);
                        qf_q = quality_function(diff_x_qf_q,diff_s_lbx_qf_q,diff_s_ubx_qf_q,P1_qf_q,P2_qf_q,P3_qf_q,alpha_qf_q);
                    }
                    
                    if (tmp_h <= sigma_qf_b*basicParam_.quality.accept_tol)
                    {
                        break;
                    }

                    if (qf_iter >= basicParam_.quality.iter_max)
                    {
                        break;
                    }
                }

                double sigma;
                if (qf_p <= qf_q)
                {
                    sigma = sigma_qf_p;

                    diff.x = diff_x_qf_p;
                    diff.lambda = diff_lambda_qf_p;
                    diff.s_lbx = diff_s_lbx_qf_p;
                    diff.s_ubx = diff_s_ubx_qf_p;
                }
                else
                {
                    sigma = sigma_qf_q;

                    diff.x = diff_x_qf_q;
                    diff.lambda = diff_lambda_qf_q;
                    diff.s_lbx = diff_s_lbx_qf_q;
                    diff.s_ubx = diff_s_ubx_qf_q;
                }
                
                barrier_.mu = sigma*dual_measurement;
                solve_flag = 1;
            }
            else 
            {
                throw std::runtime_error("undefined barrier parameter update mode");
            }

            if (basicParam_.printLevel >= 2)
            {
                printf("mu = %.16f ---> %.16f\n",mu_old,barrier_.mu);
            }

            // 计算方向(如果需要)和步长
            r_lbxs_relaxed = r_lbxs.array() - barrier_.mu;
            r_ubxs_relaxed = r_ubxs.array() - barrier_.mu;
            R_lbx.setZero();
            R_ubx.setZero();
            R_lbx(idx_lbx) = r_lbxs_relaxed.array()/v_lbx_active.array();
            R_ubx(idx_ubx) = r_ubxs_relaxed.array()/v_ubx_active.array();
            KKT_b(Eigen::seq(0,prob_.n_x - 1)) = -r_x - R_lbx + R_ubx;
            KKT_b(Eigen::lastN(prob_.n_eq)) = r_lambda;

            if (!solve_flag)
            {
                KKT_solve1(KKT_b,tmp_sol);
                
                diff.x = tmp_sol(Eigen::seq(0,prob_.n_x - 1));
                diff.lambda = tmp_sol(Eigen::lastN(prob_.n_eq));
                diff.s_lbx.setZero();
                diff.s_lbx(idx_lbx) = -R_lbx(idx_lbx) - (s_lbx_active.array()*diff.x(idx_lbx).array()/v_lbx_active.array()).matrix();
                diff.s_ubx.setZero();
                diff.s_ubx(idx_ubx) = -R_ubx(idx_ubx) + (s_ubx_active.array()*diff.x(idx_ubx).array()/v_ubx_active.array()).matrix();
            }

            double tau = std::min(basicParam_.tau_max,std::max(basicParam_.tau_min,1 - barrier_.mu));

            std::pair<double,double> alpha = get_stepLength(diff.x,diff.s_lbx,diff.s_ubx,tau);
            double alpha_primal = alpha.first;
            double alpha_dual = alpha.second;

            // 开始线搜索
            bool accept_flag = 0;
            bool watchDog_reset = 0;

            x_new = sol_.x + alpha_primal*diff.x;

            // 询问filter是否接受
            double val_f_new;
            get_val_f(x_new,p,val_f_new,1);
            get_val_eq(x_new,p,val_eq_new,1);
            double norm_eq_new = val_eq_new.lpNorm<1>();
            std::pair<double,double> pair_new = {val_f_new,norm_eq_new};
            bool filter_flag = filter_.isAcceptabel(pair_new);
            if (filter_flag == 0) // filter不接受
            {
                // 尝试merit function测试
                double merit_sigma;
                double PHI_0;
                double D_0;
                if (watchDog_.iter == 0)
                {
                    merit_sigma = std::min(basicParam_.merit_sigma_max,sol_.lambda.lpNorm<Eigen::Infinity>());

                    double val_f;
                    get_val_f(sol_.x,p,val_f,1);
                    get_val_eq(sol_.x,p,val_eq,1);
                    double norm_eq = val_eq.lpNorm<1>();
                    PHI_0 = val_f + merit_sigma*norm_eq;
                    D_0 = val_df.transpose()*diff.x - merit_sigma*norm_eq;

                    // 初始化watchDog
                    watchDog_.PHI_0 = PHI_0;
                    watchDog_.D_0 = D_0;
                    watchDog_.alpha_primal[0] = alpha_primal;
                    watchDog_.merit_sigma = merit_sigma;
                    watchDog_.pair_0 = {val_f,norm_eq};
                }
                else
                {
                    merit_sigma = watchDog_.merit_sigma;

                    PHI_0 = watchDog_.PHI_0;
                    D_0 = watchDog_.D_0;
                }

                double PHI_new = val_f_new + merit_sigma*norm_eq_new;
                double alpha_primal_soc = alpha_primal;
                if (PHI_new > PHI_0 + basicParam_.lineSearch.eta*watchDog_.alpha_primal[0]*D_0)
                {
                    // 进行二阶校正
                    bool soc_flag = 0;
                    r_lambda_soc = r_lambda;
                    KKT_b_soc = KKT_b;
                    diff_x_soc = diff.x;
                    int soc_iter = 0;
                    tmp_x = sol_.x + alpha_primal_soc*diff_x_soc;
                    get_val_eq(tmp_x,p,val_eq_soc_new,1);
                    while (basicParam_.soc_iter_max >= 1)
                    {
                        soc_iter += 1;

                        val_eq_soc = val_eq_soc_new;
                        r_lambda_soc = alpha_primal_soc*r_lambda_soc + val_eq_soc;
                        KKT_b_soc(Eigen::lastN(prob_.n_eq)) = r_lambda_soc;

                        KKT_solve1(KKT_b_soc,tmp_sol);
                        diff_x_soc = tmp_sol(Eigen::seq(0,prob_.n_x - 1));
                        
                        alpha_primal_soc = get_stepLength_xOnly(diff_x_soc,tau);
                        x_soc = sol_.x + alpha_primal_soc*diff_x_soc;

                        double val_f_soc;
                        get_val_f(x_soc,p,val_f_soc,1);
                        get_val_eq(x_soc,p,tmp_val_eq,1);
                        double norm_eq_soc = tmp_val_eq.lpNorm<1>();
                        std::pair<double,double> pair_soc = {val_f_soc,norm_eq_soc};
                        filter_flag = filter_.isAcceptabel(pair_soc);
                        if (filter_flag == 0)
                        {
                            double PHI_soc = val_f_soc + merit_sigma*norm_eq_soc;

                            if (PHI_soc > PHI_0 + basicParam_.lineSearch.eta*watchDog_.alpha_primal[0]*D_0)
                            {
                                /* do nothing */
                            }
                            else
                            {
                                soc_flag = 1;

                                break;
                            }
                        }
                        else
                        {
                            soc_flag = 1;

                            break;
                        }
                        
                        if (soc_iter >= basicParam_.soc_iter_max)
                        {
                            break;
                        }

                        tmp_x = sol_.x + alpha_primal_soc*diff_x_soc;
                        get_val_eq(tmp_x,p,val_eq_soc_new,1);
                        if (val_eq_soc_new.lpNorm<Eigen::Infinity>() > basicParam_.soc_k*val_eq_soc.lpNorm<Eigen::Infinity>())
                        {
                            break;
                        }
                    }
                    
                    if (soc_flag) // 二阶校正成功
                    {
                        if (basicParam_.printLevel >= 2)
                        {
                            printf("soc succeed for trying %d times\n",soc_iter);
                        }

                        diff.x = diff_x_soc;
                        diff.lambda = tmp_sol(Eigen::lastN(prob_.n_eq));
                        diff.s_lbx.setZero();
                        diff.s_lbx(idx_lbx) = -R_lbx(idx_lbx) 
                            - (s_lbx_active.array()*diff.x(idx_lbx).array()/v_lbx_active.array()).matrix();
                        diff.s_ubx.setZero();
                        diff.s_ubx(idx_ubx) = -R_ubx(idx_ubx)
                            + (s_ubx_active.array()*diff.x(idx_ubx).array()/v_ubx_active.array()).matrix();
                        
                        alpha_primal = alpha_primal_soc;
                        alpha_dual = get_stepLength_sOnly(diff.s_lbx,diff.s_ubx,tau);

                        accept_flag = 1;
                        watchDog_reset = 1;
                    }
                    else // 二阶校正失败
                    {
                        if (basicParam_.printLevel >= 2)
                        {
                            printf("soc failed for trying %d times\n",soc_iter);
                        }

                        // 更新watchDog
                        watchDog_.iter += 1;
                        watchDog_.start[watchDog_.iter - 1] = sol_;
                        watchDog_.steps[watchDog_.iter - 1] = diff;
                        watchDog_.alpha_primal[watchDog_.iter - 1] = alpha_primal;
                        watchDog_.alpha_dual[watchDog_.iter - 1] = alpha_dual;
                        watchDog_.PHI[watchDog_.iter - 1] = PHI_new;
                        watchDog_.pairs[watchDog_.iter - 1] = pair_new;

                        if (watchDog_.iter == 1 && basicParam_.printLevel >= 2)
                        {
                            printf("watchDog started\n");
                        }
                        
                        // watchDog达到了最大次数，或迭代达到最大次数
                        if (watchDog_.iter > watchDog_.iter_max || iter >= basicParam_.iter_max)
                        {
                            // 寻找合适的起点
                            int tmp_ptr = watchDog_.iter - 1;
                            while (1)
                            {
                                if (watchDog_.pairs[tmp_ptr].first <= watchDog_.pair_0.first - filter_.beta*watchDog_.pair_0.second)
                                {
                                    break;
                                }

                                if (watchDog_.pairs[tmp_ptr].second <= (1 - filter_.beta)*watchDog_.pair_0.second)
                                {
                                    break;
                                }
                                
                                if (watchDog_.PHI[tmp_ptr] < watchDog_.PHI_0)
                                {
                                    break;
                                }
                                
                                tmp_ptr -= 1;
                                if (tmp_ptr <= 0)
                                {
                                    tmp_ptr = 0;

                                    break;
                                }
                            }

                            if (basicParam_.printLevel >= 2)
                            {
                                printf("watchDog failed, armijo section start at %dth iteration\n",iter + tmp_ptr - watchDog_.iter + 1);
                            }
                            
                            alpha_primal = watchDog_.alpha_primal[tmp_ptr];
                            alpha_dual = watchDog_.alpha_dual[tmp_ptr];

                            PHI_new = watchDog_.PHI[tmp_ptr];

                            if (tmp_ptr > 0)
                            {
                                PHI_0 = watchDog_.PHI[tmp_ptr - 1];
                            }
                            else
                            {
                                PHI_0 = watchDog_.PHI_0;
                            }
                            
                            sol_.x = watchDog_.start[tmp_ptr].x;
                            sol_.lambda = watchDog_.start[tmp_ptr].lambda;
                            sol_.s_lbx = watchDog_.start[tmp_ptr].s_lbx;
                            sol_.s_ubx = watchDog_.start[tmp_ptr].s_ubx;

                            diff.x = watchDog_.steps[tmp_ptr].x;
                            diff.lambda = watchDog_.steps[tmp_ptr].lambda;
                            diff.s_lbx = watchDog_.steps[tmp_ptr].s_lbx;
                            diff.s_ubx = watchDog_.steps[tmp_ptr].s_ubx;

                            // armijo线搜索
                            Eigen::VectorXd tmp_vector = Eigen::VectorXd::Constant(basicParam_.lineSearch.iter_max + 1,inf_);
                            tmp_vector(0) = PHI_new;

                            int m = 1;
                            while (1)
                            {
                                double m_alpha = alpha_primal*std::pow(basicParam_.lineSearch.beta,m);

                                x_new = sol_.x + m_alpha*diff.x;

                                get_val_f(x_new,p,val_f_new,1);
                                get_val_eq(x_new,p,val_eq_new,1);
                                norm_eq_new = val_eq_new.lpNorm<1>();
                                pair_new = {val_f_new,norm_eq_new};

                                filter_flag = filter_.isAcceptabel(pair_new);
                                if (filter_flag == 0) // filter不接受
                                {
                                    PHI_new = val_f_new + merit_sigma*norm_eq_new;
                                    tmp_vector(m) = PHI_new;

                                    // merit function接受
                                    if (PHI_new <= PHI_0 + basicParam_.lineSearch.eta*watchDog_.alpha_primal[0]*D_0)
                                    {
                                        break;
                                    }
                                }
                                else // filter接受
                                {
                                    break;
                                }
                                
                                m += 1;
                                if (m > basicParam_.lineSearch.iter_max)
                                {
                                    Eigen::Index minPtr;
                                    tmp_vector.minCoeff(&minPtr);
                                    m = minPtr;

                                    break;
                                }
                            }

                            // 更新
                            double m_alpha = alpha_primal*std::pow(basicParam_.lineSearch.beta,m);
                            sol_.x += m_alpha*diff.x;
                            sol_.lambda += m_alpha*diff.lambda;
                            sol_.s_lbx += alpha_dual*diff.s_lbx;
                            sol_.s_ubx += alpha_dual*diff.s_ubx;

                            // 重置watchDog
                            watchDog_.refresh();
                        }
                        else // watchDog 没有达到最大次数
                        {
                            accept_flag = 1;
                        }
                    }
                }
                else // merit function接受
                {
                    accept_flag = 1;
                    watchDog_reset = 1;
                }
            }
            else // filter接受
            {
                accept_flag = 1;
                watchDog_reset = 1;
            }
            
            // 线搜索结束后更新变量并根据搜索结果调整一些东西
            if (accept_flag == 1)
            {
                sol_.x += alpha_primal*diff.x;
                sol_.lambda += alpha_primal*diff.lambda;
                sol_.s_lbx += alpha_dual*diff.s_lbx;
                sol_.s_ubx += alpha_dual*diff.s_ubx;
            }

            if (watchDog_.iter > 0 && watchDog_reset == 1)
            {
                // 重置看门狗
                watchDog_.refresh();

                if (basicParam_.printLevel >= 2)
                {
                    printf("watchDog succeed\n");
                }
            }
            
            // 根据当前迭代确定罚参数更新模式
            if (accept_flag && watchDog_reset && barrier_.isProtected)
            {
                barrier_.isProtected = 0;

                if (basicParam_.printLevel >= 2)
                {
                    std::cout<<"barrier update mode back into ";
                    if (barrier_.updateMode_ == barrier::updateMode::Monotone)
                    {
                        std::cout<<"Monotone";
                    }
                    else if (barrier_.updateMode_ == barrier::updateMode::LOQO_rule)
                    {
                        std::cout<<"LOQO_rule";
                    }
                    else if (barrier_.updateMode_ == barrier::updateMode::QualityFunction)
                    {
                        std::cout<<"QualityFunction";
                    }
                    std::cout<<std::endl;
                }
            }
            else if (watchDog_.iter > 0 && barrier_.isProtected == 0 && homotopy_.isDone == 1 && barrier_.updateMode_ != barrier::updateMode::Monotone)
            {
                barrier_.isProtected = 1;

                mu_old = barrier_.mu;
                barrier_.mu = barrier_.k*dual_measurement;

                if (basicParam_.printLevel >= 2)
                {
                    printf("barrier update protected:\n mu = %.16f ---> %.16f\n",mu_old,barrier_.mu);
                }
            }
        }

        sol_KKT_error = KKT_error;

        if (isQPprob)
        {
            Eigen::VectorXd tmpV = sol_.x.topRows(x0.size());
            sol_.x = tmpV;
        }
        
        //
        cpuTime = tmp_clock.toc(0);
        if (basicParam_.printLevel >= 1)
        {
            printf("\n====================================================================\n");
            if (exit_flag == 0)
            {
                printf("  solver stops because it reaches the maximum number of iterations");
            }
            else if (exit_flag == 1)
            {
                printf("                      optimal solution found");
            }
            printf("\n====================================================================\n\n");

            showProcess();
        }

        // 输出
        refresh_flag = false;
        return sol_;
    }
} // namespace homo_ocp
