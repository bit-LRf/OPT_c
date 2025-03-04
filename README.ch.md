# homo_ocp

## 介绍
基于内点法非线性求解器，内置参数同伦路径跟踪温启动，大部分情况下对于高频的MPC问题能够提高计算速度

## 需求
推荐系统：`Ubuntu24.04`

### C++ 环境：  
1.  `C++17`及以上
2.  `eigen`: 3.4.0及以上
3.  `openblas`
4.  `spral`(可选，若没有则性能一般且不会检查问题的凸性，i.e.不会修正非凸问题)

#### spral库依赖：  
1.  `cuda`(可选)
2.  `metis`
3.  `hwloc`(如果使用cuda的话请添加cuda相关选项对源文件进行编译)

### matlab环境：
1.  `matlab`版本需求：2022a及以上
2.  `casadi`

## C++环境配置
**以下仅为作者的安装方法，任何能够安装eigen和spral的方式都OK，记得修改CMakeLists里的配置就行**  
**一个替代方案是从spral的[git](https://github.com/ralna/spral)库中直接下载预编译版本,缺点是性能稍差一些**  

安装过程中请使用以下指令进入root模式
```bash
su root
```
之前没进过root的话使用以下指令设置root密码然后重新进入root模式  
```bash
sudo passwd root
```

cuda第一个装,spral最后一个装，其他安装顺序随意
1.  安装`g++12,gcc12`(因为cuda编译需要这个)：apt包管理器下载
2.  安装`cuda`：去官网下载（教程太多了自己找）
3.  安装`eigen`：
把eigen从[官网](https://eigen.tuxfamily.org)中扒下来自己照着install文件装，比较简单，下面的'build_dir'是你自己命名的build文件夹名称

```bash
cd build_dir
cmake source_dir
make install
```

4.  安装`openblas`：
也是去[官网](https://www.openblas.net/)下载，或者从git上面扒也行，安装比较简单：

```bash
make
make install
```

5.  安装`hwloc`：在[hwloc官网](https://www.open-mpi.org/projects/hwloc/)上找源码安装包，然后编译的时候添加cuda支持：

```bash
./configure -with-cuda=/path/to/cuda --enable-cuda
make
make install
```
*注意：不要去git上扒，少东西*

6.  安装`metis`：
用spral给的[metis 4.0 ](https://github.com/coin-or-tools/ThirdParty-Metis)版本：

```bash
git clone https://github.com/coin-or-tools/ThirdParty-Metis.git
cd ThirdParty-Metis && ./get.Metis
mkdir build
cd build
../configure
make && make install
```
*注意：使用此方法安装的库叫做coinmetis而不是metis，找库的时候别找错了*  

7.  安装`spral`：
前面的东西都安装好了之后用apt包管理器下载meson，然后从spral的[git](https://github.com/ralna/spral)或[官网](https://ralna.github.io/spral/)把源码扒下来，修改meson_options.txt文件里的内容（注意设置cuda和metis的版本），然后：(builddir也是build文件夹名字，自己定)

```bash
# Setup SPRAL
meson setup builddir

# Compile SPRAL
meson compile -C builddir

# Install SPRAL
meson install -C builddir

# Test SPRAL
meson test -C builddir
```

环境变量设置，建议装一个设置一个：

```bash
# lib directories
export PATH=$PATH:/usr/local/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

# cuda
export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export C_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${CUDA_HOME}/include
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${CUDA_HOME}/include
export NVCC_INCLUDE_FLAGS=${NVCC_INCLUDE_FLAGS}:-I${CUDA_HOME}/include

# metis
export METISDIR=/usr/local/lib

# openblas
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/OpenBLAS/lib

# spral
export OMP_CANCELLATION=TRUE
export OMP_PROC_BIND=TRUE
```
### 测试spral的安装环境
envTest文件夹内修改CMakeLists里的文件路径然后编译运行

## matlab环境配置
matlab环境配置比较简单，去casadi[官网](https://web.casadi.org/)把对应matlab版本的casadi压缩包下载下来然后解压，在matlab的路径中添加casadi就行了

## 安装

```bash
cd develop
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
make test # 如果编译了测试用例
make install
```
如果没有安装spral库，则cmake命令添加以下参数以安装仅使用Eigen的版本：
```bash
-DOPT_BUILD_WITH_SPRAL=OFF
```
*警告：仅使用Eigen的版本不仅性能更差，并且不会修正非凸问题*  

如果不希望编译测试用例，则在cmake命令添加以下参数：
```bash
-DOPT_BUILD_TEST=OFF
```

## 基本使用方法
1. 添加头文件`homo_ocp.h`
*注意：在引用任何Eigen库头文件前引用homo_ocp的头文件*
2. 如果编译时使用了spral库，则定义宏`HOMO_OCP_USE_SPRAL`，如果编译时仅使用Eigen，
则定义宏`HOMO_OCP_USE_EIGEN`  
3. 在CMakeLists里添加`find_library(HOMO_OCP_LIBRARY homo_ocp)`  
4. 使用casadi建立如下形式的标准NLP问题：  
$
minimize:   f(x)  \quad\\
s.t.      \quad\\  
eq(x) = 0  \quad\\
ineq(x) \geq 0  \quad\\
lbx \leq x \leq ubx  \quad\\
$
在建立问题时，x的上下界可暂时先不建立
5. 使用`casadi_nlpProbGen`生成`.h`和`.cpp`文件，具体可以参考matlab文件large_prob_gen.m和simple_prob_gen.m的问题生成案例 
6. 将生成的文件包含在你的代码目录中，添加`.h`头文件，使用宏`HOMO_OCP_PROB`实例化NLP问题，设置参数和上下界
7. 实例化求解器`homo_ocp::solver`并将前面的NLP问题结构体传入求解器，调用`.solve()`完成求解  

*具体可以参考`homo_ocp_mainTest.cpp`中的做法*

**note: 如果同时调用多个求解器发现性能有问题时可以检查CPU的资源占用情况，并手动绑定进程的CPU核心，参考下面的命令行：**  
```bash
taskset -c 5,6,7,8 ./test # 将test进程绑定到5,6,7,8号CPU核心
taskset -c 5-8 ./test # 与上面的效果相同
```

## 函数说明

### matlab
1. `casadi_nlpProbGen(x,p,f,eq,ineq,funName，suffix)`:此函数用于生成标准NLP问题，使用方法参考`simple_prob_gen`和`large_prob_gen`
2. `casadi_qpProbGen(H,c,A_eq,b_eq,A_ineq,b_ineq,funName,suffix)`：此函数用于生成QP问题的NLP描述形式，主要是帮助简化问题构建难度

### C++
**以下描述时省略命名空间`homo_ocp`**

#### 实例化非线性优化求解器
1. 使用宏`HOMO_OCP_PROB(PROB_NAME,FUN_NAME)`：用来初始化求解问题结构体，`FUN_NAME`应当与前面matlab中的`funName`保持一致
2. `solver(prob PROB_NAME, const double what_is_inf, const int cpu_core, const int printLevel)`：核心功能，用于实例化一个求解器。   
$\quad$ what_is_inf：当边界大于等于此数值时认为没有约束  
$\quad$ cpu_core：openMP分配的cpu线程  
$\quad$ printLevel：打印级别：0：不打印；1：只打印最终结果；2：打印全过程

#### 使用二次规划接口进行实例化
定义如下二次规划问题：  
$
minize: 0.5x^THx + c^Tx  \quad\\
s.t.  \quad\\
A_{eq}x = b_{eq}  \quad\\
A_{ineq}x \geq b_{ineq}  \quad\\
lbx \leq x \leq ubx  \quad\\
$
一个较为简单的实例化方式是使用：  
`solver::solver(
    SpMat &H, Eigen::VectorXd &c, 
    SpMat &A_eq, Eigen::VectorXd &b_eq, 
    SpMat &A_ineq, Eigen::VectorXd &b_ineq, 
    Eigen::VectorXd &lbx, Eigen::VectorXd &ubx,
    const double what_is_inf, const int cpu_core, const int printLevel
)` 
接口。

#### 参数修改：
在实例化`solver`后直接修改成员里的参数就行了

```cpp
struct basicParam
    {
        int iter_max = 1e2; // 最大迭代次数
        double accept_tol = 1e-6; // KKT系统容差，达到容差后认为收敛
        double small_tol = 1e-12; // 无穷小，小于该值认为等于0
        double kappa_1 = 0.01; // 初始值到单边界的距离系数
        double kappa_2 = 0.01; // 初始值到双边界的距离系数
        double lambda_max = 1e4; // 最大等式乘子
        double tau_min = 0.9; // 最小步长因子
        double tau_max = 1 - 1e-6; // 最大步长因子
        double merit_sigma_max = 10; // 最大优势函数罚参
        int soc_iter_max = 2; // 二阶校正最大允许次
        double soc_k = 0.99; // 二阶校正新的等式误差最小下降为上次的soc_k倍，否则退出二阶校正
        int printLevel = 2; // 0：不打印消息；>=1：打印结果消息；>=2：打印过程消息
        bool useHomotopy = false; // 是否使用参数同伦进行温启动
        
        struct monotone
        {
            double k = 0.2; // 罚参数更新参数
            double t = 1.5; // 罚参数更新参数
            double epsilon = 10; // 罚参数更新条件参数
        } monotone;

        struct loqo
        {
            double k = 0.05; // 罚参数更新参数
            double t = 2; // 罚参数更新参数
            double alpha = 0.1; // 罚参数更新参数
            double mu_min = 1e-9; // 最小罚参数
        } loqo;
        
        struct quality
        {
            double epsilon = 0.01; // 与确定区间有关的参数，需要比较小
            int iter_max = 12; // 黄金分割法最大迭代次数
            int block = 6; // 搜索区间划分个数
            double accept_tol = 1e-2; // 黄金分割容许相对误差
            double sigma_max = 1e3;
            double sigma_min = 1e-8;
            double mu_min = 1e-9;
        } quality;
        
        struct lineSearch
        {
            int iter_max = 8; // 线搜索最大允许次数
            double beta = 0.7; // 参数，范围（0，1）
            double eta = 0.1; // 参数，范围（0，1）
        } lineSearch;
    };

    struct ICor
    {
        double w_ac = 10; // w的上升速度
        double a_ac = 10; // a的上升速度
        double w_dc = 2; // w的下降速度
        double a_dc = 2; // a的下降速度
        double w_0 = 1e-2; // 初始值
        double a_0 = 1e-6; // 初始值
        double w_min = 1e-4;
        double a_min = 1e-8;
        double w_max = 1e4;
        double a_max = 1e4;
        int switchIter = 3; // 模式切换的迭代次数
        int porbeIter = 3; // 在修正模式下试图回到一般模式的试探频率
    };

    struct barrier
    {
        enum class updateMode
        {
            Monotone,
            LOQO_rule,
            QualityFunction,
        };

        double mu_max = 1e3; // 最大罚参数
        double mu_0 = 0.1; // 初始罚参数
        updateMode updateMode_ = updateMode::QualityFunction;
            // 更新模式，包括：
            // Monotone：单调模式，参考IPOPT
            // LOQO_rule：参考LOQO中的启发式罚参数更新模式
            // Quality_function：以指标函数确定新的罚参数
        double k = 0.8; // 保护模式开启时重置mu有关的参数
    };

    struct watchDog
    {
        int iter_max = 4; // 看门狗最大允许非单调次数
    };

    struct homotopy
    {
        double diff_lota_min = 1.0; // 最小同伦步长
        double diff_lota_max = 1.0; // 最大同伦步长
        double epsilon = 1; // 单调更新参数
        int iter_max = 100;
    };
```

#### 一些`homo_ocp::solver`的成员函数：

```cpp
sol solve(const Eigen::VectorXd& x0) //使用初值x0进行求解，当有温启动信息时x0无作用  
void refresh(); // 刷新过程量
void reLoadParam(const Eigen::VectorXd& p); // 修改问题的参数
void reLoadBoundary(const Eigen::VectorXd *lbx, const Eigen::VectorXd *ubx, const bool check); //修改问题的边界
void setWsInfo(const sol &m_sol, const Eigen::VectorXd &p); // 使用其他已知的温启动信息
void clearWsInfo(); // 将同伦温启动信息清除
void reLoadQP(const Eigen::VectorXd *c, const Eigen::VectorXd *b_eq, const Eigen::VectorXd *b_ineq);// 快速变换QP问题的部分参数
void reLoadQPboundary(const Eigen::VectorXd *lbx, const Eigen::VectorXd *ubx, const bool check);// 修改QP问题的边界，作用和reLoadBoundary差不多
void showProcess(); // 打印最终结果
sol getSolution(); // 输出求解结果
int getIteration(); // 输出迭代次数
int getHomotopyConvergeIteration(); // 输出同伦路径跟踪阶段的迭代次数
processRecorder getProcessInfo(); // 输出过程记录信息
int getExitFlag(); // 退出标志：0：达到最大迭代次数；1：收敛
wsInfo getWsInfo(); // 取出温启动结构体 
```
