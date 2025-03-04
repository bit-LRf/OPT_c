cmake_minimum_required(VERSION 3.10)

# 
project(homo_ocp VERSION 1.0 LANGUAGES CXX)

# 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# specific your options 
option(OPT_BUILD_WITH_SPRAL "Enable SPRAL" ON)
option(OPT_BUILD_TEST "test" ON)

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

# 头文件
include_directories(include)
file(GLOB_RECURSE HOMO_OCP_HEADERS "include/*")

# 源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 库文件
add_library(homo_ocp SHARED 
    ${SOURCES}
)
target_link_libraries(homo_ocp
    ${OpenBLAS_LIBRARIES}
)

# 测试文件
if (OPT_BUILD_TEST)
    file(GLOB_RECURSE HOMO_OCP_TEST_SRC "test/*.cpp")

    add_executable(homo_ocp_test 
        ${SOURCES}
        ${HOMO_OCP_TEST_SRC}
    )
    target_include_directories(homo_ocp_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )
    target_link_libraries(homo_ocp_test 
        ${OpenBLAS_LIBRARIES} 
    )

    enable_testing()
    add_test(NAME TEST COMMAND homo_ocp_test)

    message("=====build homo_ocp_test=====")
endif ()

# SPRAL依赖
if (OPT_BUILD_WITH_SPRAL)
    add_definitions(-DHOMO_OCP_USE_SPRAL)
    find_library(SPRAL_LIBRARY spral)

    if (OPT_BUILD_TEST)
        target_link_libraries(homo_ocp_test 
            ${SPRAL_LIBRARY}
        )
    endif ()
    
    target_link_libraries(homo_ocp
        ${SPRAL_LIBRARY}
    )

    message("=====build homo_ocp with spral=====")
else ()
    add_definitions(-DHOMO_OCP_USE_EIGEN)

    message("=====build homo_ocp with eigen only=====")
endif ()

# install
install(TARGETS homo_ocp DESTINATION lib)
install(FILES ${HOMO_OCP_HEADERS} DESTINATION include)