# homo_ocp

This document was translated by AI. 

This repository is copied from [https://gitee.com/bit_liruifeng/ocp_opt.git](https://gitee.com/bit_liruifeng/ocp_opt.git) repository from time to time. To get the latest code, please go to the original repository.

## Introduction
A nonlinear solver based on the interior-point method, with built-in parameter homotopy path tracking for warm start. In most cases, it can improve the computational speed for high-frequency MPC problems.

## Requirements
Recommended system: `Ubuntu 24.04`

### C++ Environment:
1. `C++17` or higher
2. `eigen`: version 3.4.0 or higher
3. `openblas`
4. `spral` (optional; without it, performance is average and the solver will not check for convexity, i.e., it will not correct non-convex problems)

#### Dependencies for spral library:
1. `cuda` (optional)
2. `metis`
3. `hwloc` (If using CUDA, add CUDA-related options when compiling the source files)

### MATLAB Environment:
1. MATLAB version: 2022a or higher
2. `casadi`

---

## C++ Environment Setup
**The following is the author's installation method. Any method that can install eigen and spral is acceptable, as long as you modify the configurations in `CMakeLists.txt` accordingly.**  
**An alternative is to download the precompiled version from the spral [git](https://github.com/ralna/spral) repository, but the performance is slightly worse.**

Enter root mode during the installation process using the following command:
```bash
su root
```
If you have not entered root mode before, set the root password and re-enter root mode using:
```bash
sudo passwd root
```

Install CUDA first and spral last; the order of other installations is arbitrary.

1. **Install `g++12` and `gcc12`** (required for CUDA compilation): Use the apt package manager.
2. **Install `cuda`**: Download from the official website (there are many tutorials available).
3. **Install `eigen`**: Download from the [official website](https://eigen.tuxfamily.org) and follow the installation instructions in the `install` file. Below, `build_dir` is the name of the build directory you create.

    ```bash
    cd build_dir
    cmake source_dir
    make install
    ```

4. **Install `openblas`**: Download from the [official website](https://www.openblas.net/) or from the git repository. The installation is straightforward:

    ```bash
    make
    make install
    ```

5. **Install `hwloc`**: Download the source code package from the [hwloc official website](https://www.open-mpi.org/projects/hwloc/) and compile it with CUDA support:

    ```bash
    ./configure --with-cuda=/path/to/cuda --enable-cuda
    make
    make install
    ```
    *Note: Do not download from git, as it may be incomplete.*

6. **Install `metis`**: Use the [metis 4.0](https://github.com/coin-or-tools/ThirdParty-Metis) version provided by spral:

    ```bash
    git clone https://github.com/coin-or-tools/ThirdParty-Metis.git
    cd ThirdParty-Metis && ./get.Metis
    mkdir build
    cd build
    ../configure
    make && make install
    ```
    *Note: The library installed using this method is named `coinmetis`, not `metis`. Be careful when searching for the library.*

7. **Install `spral`**: After installing the above components, download meson using the apt package manager. Then, obtain the source code from the spral [git](https://github.com/ralna/spral) repository or the [official website](https://ralna.github.io/spral/). Modify the `meson_options.txt` file to set the versions of CUDA and metis. Then:

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

Environment variable settings (it is recommended to set one after installing each component):

```bash
# Library directories
export PATH=$PATH:/usr/local/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

# CUDA
export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export C_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${CUDA_HOME}/include
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${CUDA_HOME}/include
export NVCC_INCLUDE_FLAGS=${NVCC_INCLUDE_FLAGS}:-I${CUDA_HOME}/include

# METIS
export METISDIR=/usr/local/lib

# OpenBLAS
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/OpenBLAS/lib

# SPRAL
export OMP_CANCELLATION=TRUE
export OMP_PROC_BIND=TRUE
```

### Testing the SPRAL Installation Environment
Modify the file paths in the `CMakeLists.txt` file in the `envTest` folder, then compile and run.

---

## MATLAB Environment Setup
Setting up the MATLAB environment is relatively simple. Download the CasADi package corresponding to your MATLAB version from the [CasADi official website](https://web.casadi.org/), extract it, and add the CasADi path to MATLAB.

---

## Installation

```bash
cd develop
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
make test # If test cases are compiled
make install
```

If the spral library is not installed, add the following parameter to the cmake command to install the version using only Eigen:
```bash
-DOPT_BUILD_WITH_SPRAL=OFF
```
*Warning: The version using only Eigen has worse performance and will not correct non-convex problems.*

If you do not want to compile test cases, add the following parameter to the cmake command:
```bash
-DOPT_BUILD_TEST=OFF
```

---

## Basic Usage
1. Include the header file `homo_ocp.h`.
   *Note: Include the `homo_ocp` header file before any Eigen library header files.*
2. If the spral library was used during compilation, define the macro `HOMO_OCP_USE_SPRAL`. If only Eigen was used, define the macro `HOMO_OCP_USE_EIGEN`.
3. Add `find_library(HOMO_OCP_LIBRARY homo_ocp)` in `CMakeLists.txt`.
4. Formulate the standard NLP problem using CasADi in the following form:
$$
\begin{aligned}
\text{minimize:} \quad & f(x) \\
\text{subject to:} \quad & \\
& \text{eq}(x) = 0 \\
& \text{ineq}(x) \geq 0 \\
& \text{lbx} \leq x \leq \text{ubx}
\end{aligned}
$$
The bounds of \(x\) can be temporarily omitted when formulating the problem.
5. Use `casadi_nlpProbGen` to generate `.h` and `.cpp` files. For specific examples, refer to the problem generation cases in the MATLAB files `large_prob_gen.m` and `simple_prob_gen.m`.
6. Include the generated files in your code directory, add the `.h` header file, instantiate the NLP problem using the macro `HOMO_OCP_PROB`, and set the parameters and bounds.
7. Instantiate the solver `homo_ocp::solver` and pass the NLP problem structure to the solver. Call `.solve()` to complete the solution process.
   *For specific examples, refer to the implementation in `homo_ocp_mainTest.cpp`.*

**Note: If performance issues are encountered when calling multiple solvers simultaneously, check the CPU resource usage and manually bind the process to specific CPU cores using the following command:**
```bash
taskset -c 5,6,7,8 ./test # Bind the test process to CPU cores 5, 6, 7, and 8
taskset -c 5-8 ./test # Same as above
```

---

## Function Description

### MATLAB
1. `casadi_nlpProbGen(x, p, f, eq, ineq, funName, suffix)`: This function is used to generate a standard NLP problem. For usage, refer to `simple_prob_gen` and `large_prob_gen`.
2. `casadi_qpProbGen(H, c, A_eq, b_eq, A_ineq, b_ineq, funName, suffix)`: This function generates the NLP description form of a QP problem, primarily to simplify problem formulation.

### C++  
**The following description omits the namespace `homo_ocp`.**  

#### Instantiating a Nonlinear Optimization Solver  
1. Use the macro `HOMO_OCP_PROB(PROB_NAME, FUN_NAME)`: This is used to initialize the problem structure. `FUN_NAME` should be consistent with the `funName` in the previous MATLAB code.  
2. `solver(prob PROB_NAME, const double what_is_inf, const int cpu_core, const int printLevel)`: This is the core function used to instantiate a solver.

$\quad$ what_is_inf：When the bound is greater than or equal to this value, no constraint is considered  
$\quad$ cpu_core：openMP allocated cpu threads  
$\quad$ printLevel：Print level: 0: no print; 1: only print the final result; 2: printing the whole process

See the header file for more details on the parameters.