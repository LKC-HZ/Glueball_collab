cmake_minimum_required(VERSION 3.18)
project(color_wf LANGUAGES Fortran)

# 编译器（你只在 node01 上跑的话可以写死）
set(CMAKE_Fortran_COMPILER gfortran)

# 编译选项
set(CMAKE_Fortran_FLAGS "-O2 -ffpe-summary=none")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# 可执行文件
add_executable(color_wf color_wf.f90)

# 库路径
link_directories(/usr/local/lib)

# 链接库
target_link_libraries(color_wf
    lapack
    blas
    arpack
)


