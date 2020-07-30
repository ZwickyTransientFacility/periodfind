cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

include(GNUInstallDirs)

project(periodfind LANGUAGES CXX CUDA)

add_library(periodfind SHARED
            periodfind/cuda/ce.cu)

target_compile_features(periodfind PUBLIC cxx_std_14)

set_target_properties(periodfind
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(periodfind PROPERTIES PUBLIC_HEADER
                      periodfind/cuda/ce.h)

install(TARGETS periodfind
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/periodfind)