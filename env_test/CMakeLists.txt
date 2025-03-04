cmake_minimum_required(VERSION 3.10)

# 
project(env_test VERSION 1.0 LANGUAGES CXX)

# 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# openMP settings
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# OPENBLAS
find_package(OpenBLAS REQUIRED)
include_directories(${OpenBLAS_INCLUDE_DIRS})

# SPRAL
find_library(SPRAL_LIBRARY spral)

# 
add_executable(env_test 
    src/env_test.cpp
)
target_link_libraries(env_test 
    ${SPRAL_LIBRARY}
    ${OpenBLAS_LIBRARIES} 
)


