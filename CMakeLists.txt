cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

include(GNUInstallDirs)

project(periodfind LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

add_library(periodfind SHARED
            periodfind/cuda/ce.cu
            periodfind/cuda/aov.cu
            periodfind/cuda/ls.cu)

file(GLOB_RECURSE HEADER_FILES "periodfind/cuda/*.h")

target_compile_features(periodfind PUBLIC cxx_std_14)

set_target_properties(periodfind
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(periodfind PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_link_libraries(periodfind PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(periodfind PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_compile_options(periodfind PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)

install(TARGETS periodfind
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/periodfind)
