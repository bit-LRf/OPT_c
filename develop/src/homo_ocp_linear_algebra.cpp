# include "homo_ocp.h"

namespace homo_ocp
{
    std::pair<double,double> solver::get_stepLength(
        const Eigen::VectorXd& diff_x, const Eigen::VectorXd& diff_s_lbx, const Eigen::VectorXd& diff_s_ubx, const double tau) // 警告：禁止多线程调用此函数
    {
        std::pair<double,double> out = {0,0};
        double alpha_primal = 1;
        double alpha_dual = 1;
   
        tmp_nlbx = -(sol_.x(idx_lbx) - prob_.lbx(idx_lbx)).array()/diff_x(idx_lbx).array();
        tmp_nubx = (-sol_.x(idx_ubx) + prob_.ubx(idx_ubx)).array()/diff_x(idx_ubx).array();

        if (tmp_nlbx.size() > 0)
        {
            Eigen::Index minPtr1;
            (1/tmp_nlbx.array()).maxCoeff(&minPtr1);
            if (tmp_nlbx(minPtr1) > 0)
            {
                alpha_primal = std::min(alpha_primal,tau*tmp_nlbx(minPtr1));
            }
            tmp_nlbx = -sol_.s_lbx(idx_lbx).array()/diff_s_lbx(idx_lbx).array();
            (1/tmp_nlbx.array()).maxCoeff(&minPtr1);
            if (tmp_nlbx(minPtr1) > 0)
            {
                alpha_dual = std::min(alpha_dual,tau*tmp_nlbx(minPtr1));
            }
        }
        
        if (tmp_nubx.size() > 0)
        {
            Eigen::Index minPtr2;
            (1/tmp_nubx.array()).maxCoeff(&minPtr2);
            if (tmp_nubx(minPtr2) > 0)
            {
                alpha_primal = std::min(alpha_primal,tau*tmp_nubx(minPtr2));
            }
            tmp_nubx = -sol_.s_ubx(idx_ubx).array()/diff_s_ubx(idx_ubx).array();
            (1/tmp_nubx.array()).maxCoeff(&minPtr2);
            if (tmp_nubx(minPtr2) > 0)
            {
                alpha_dual = std::min(alpha_dual,tau*tmp_nubx(minPtr2));
            }
        }

        out.first = alpha_primal;
        out.second = alpha_dual;
        return out;
    }

    double solver::get_stepLength_xOnly(const Eigen::VectorXd& diff_x, const double tau) // 警告：禁止多线程调用此函数
    {
        double alpha_primal = 1;

        if (tmp_nlbx.size() > 0)
        {
            Eigen::Index minPtr1;
            tmp_nlbx = -(sol_.x(idx_lbx) - prob_.lbx(idx_lbx)).array()/diff_x(idx_lbx).array();
            (1/tmp_nlbx.array()).maxCoeff(&minPtr1);
            if (tmp_nlbx(minPtr1) > 0)
            {
                alpha_primal = std::min(alpha_primal,tau*tmp_nlbx(minPtr1));
            }
        }
        
        if (tmp_nubx.size() > 0)
        {
            Eigen::Index minPtr2;
            tmp_nubx = (-sol_.x(idx_ubx) + prob_.ubx(idx_ubx)).array()/diff_x(idx_ubx).array();
            (1/tmp_nubx.array()).maxCoeff(&minPtr2);
            if (tmp_nubx(minPtr2) > 0)
            {
                alpha_primal = std::min(alpha_primal,tau*tmp_nubx(minPtr2));
            } 
        }

        return alpha_primal;
    }

    double solver::get_stepLength_sOnly(const Eigen::VectorXd& diff_s_lbx, const Eigen::VectorXd& diff_s_ubx, const double tau) // 警告：禁止多线程调用此函数
    {
        double alpha_dual = 1;

        if (tmp_nlbx.size() > 0)
        {
            Eigen::Index minPtr1;
            tmp_nlbx = -sol_.s_lbx(idx_lbx).array()/diff_s_lbx(idx_lbx).array();
            (1/tmp_nlbx.array()).maxCoeff(&minPtr1);
            if (tmp_nlbx(minPtr1) > 0)
            {
                alpha_dual = std::min(alpha_dual,tau*tmp_nlbx(minPtr1));
            }
        }
        
        if (tmp_nubx.size() > 0)
        {
            Eigen::Index minPtr2;
            tmp_nubx = -sol_.s_ubx(idx_ubx).array()/diff_s_ubx(idx_ubx).array();
            (1/tmp_nubx.array()).maxCoeff(&minPtr2);
            if (tmp_nubx(minPtr2) > 0)
            {
                alpha_dual = std::min(alpha_dual,tau*tmp_nubx(minPtr2));
            }
        }

        return alpha_dual;
    }

    double solver::quality_function(
        const Eigen::VectorXd& diff_x, const Eigen::VectorXd& diff_s_lbx, const Eigen::VectorXd& diff_s_ubx,
        const Eigen::VectorXd& P1, const Eigen::VectorXd& P2, const Eigen::VectorXd& P3, const std::pair<double,double> alpha)
    {
        double alpha_primal = alpha.first;
        double alpha_dual = alpha.second;

        return std::max({
            (r_x + alpha_primal*P1 - alpha_primal*P3
            - alpha_dual*diff_s_lbx + alpha_dual*diff_s_ubx).lpNorm<Eigen::Infinity>(),

            (r_lambda + alpha_primal*P2).lpNorm<Eigen::Infinity>(),

            ((s_lbx_active + alpha_dual*diff_s_lbx(idx_lbx)).array()*
            (sol_.x(idx_lbx) + alpha_primal*diff_x(idx_lbx) - prob_.lbx(idx_lbx)).array()).matrix().lpNorm<Eigen::Infinity>(),

            ((s_ubx_active + alpha_dual*diff_s_ubx(idx_ubx)).array()*
            (-(sol_.x(idx_ubx) + alpha_primal*diff_x(idx_ubx)) + prob_.ubx(idx_ubx)).array()).matrix().lpNorm<Eigen::Infinity>()
        });
    }

    bool filter::isAcceptabel(std::pair<double,double> pair_new)
    {
        std::vector<std::pair<double,double>> tmp_pairs;
        int n = pairs.size();

        bool flag = 1;
        for (size_t i = 0; i < n; i++)
        {
            if (pair_new.first <= pairs[i].first - beta*pairs[i].second || pair_new.second <= (1 - beta)*pairs[i].second)
            {
                if (pair_new.first <= pairs[i].first - beta*pairs[i].second && pair_new.second <= (1 - beta)*pairs[i].second)
                {
                    /* do nothing */
                }
                else
                {
                    tmp_pairs.push_back({pairs[i].first,pairs[i].second});
                }
            }
            else
            {
                flag = 0;

                break;
            }
        }
        
        if (flag == 1)
        {
            tmp_pairs.push_back(pair_new);
            pairs = tmp_pairs;
        }

        return flag;
    }
} // namespace homo_ocp
