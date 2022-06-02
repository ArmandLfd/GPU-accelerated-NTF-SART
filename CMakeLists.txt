#------------------------------------------------------------------------------ -
#
#Copyright © 2021 Université Libre de Bruxelles (ULB)
#Copyright © 2021 Universidad Politécnica de Madrid (UPM)
#
#Authors : Jaime Sancho, Daniele Bonatto, Laurie Van Bogaert
#Contact : Gauthier.Lafruit@ulb.ac.be
#
#Plane Sweep for INFO-H-503 CUDA project
#This file is used to generate buildable versions of Plane Sweep
#
#Permission is hereby granted, free of charge, to the members of the Moving Picture
#Experts Group (MPEG) obtaining a copy of this software and associated documentation
#files (the "Software"), to use the Software exclusively within the framework of the
#MPEG - I (immersive) and MPEG - I Visual activities, for the sole purpose of
#developing the MPEG - I standard.This permission includes without limitation the
#rights to use, copy, modify and merge copies of the Software, and explicitly
#excludes the rights to publish, distribute, sublicense, sell, embed into a product
#or a service and/or otherwise commercially exploit copies of the Software without
#the written consent of the owner (UPM, ULB).
#
#This permission is provided subject to the following conditions :
#The above copyright notice and this permission notice shall be included in all
#copies, substantial portions or derivative works of this file.
#
#------------------------------------------------------------------------------ -*/


cmake_minimum_required(VERSION 3.17)

set(PROJECT_NAME "TensorLayer")
project(${PROJECT_NAME} LANGUAGES CUDA CXX)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source build detected!")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package( OpenCV REQUIRED)


find_package(CUDA REQUIRED)         # This works for me
include(CheckLanguage)
check_language(CUDA)


if(${CMAKE_CUDA_COMPILER} STREQUAL "")
    message(FATAL_ERROR "CUDA COMPILER not found!")
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    set(CMAKE_CXX_EXTENSIONS OFF) 
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()


#option(WITH_OPENMP "Use OpenMP" ON)
#set(OPENMP_LIB "")
#if (WITH_OPENMP)
#	find_package(OpenMP REQUIRED)
#	set(OPENMP_LIB optimized OpenMP::OpenMP_CXX)
#endif(WITH_OPENMP)


#if(MSVC)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
#else()
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp -Wextra -Wpedantic -std=c++11")
#endif()

file(GLOB kernel_tensor_layer_headers
    ./kernels/*.cuh
    )

file(GLOB kernel_tensor_layer_src
    ./kernels/*.cu
    )

source_group(kernel_headers FILES ${kernel_tensor_layer_headers})
source_group(kernel_sources FILES ${kernel_tensor_layer_src})

add_library(${PROJECT_NAME}CUDA ${kernel_tensor_layer_headers} ${kernel_tensor_layer_src})
set_property(TARGET ${PROJECT_NAME}CUDA PROPERTY
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
                )
target_compile_options(${PROJECT_NAME}CUDA PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-lineinfo;--source-in-ptx;>")



target_include_directories(${PROJECT_NAME}CUDA PUBLIC
                ${CUDA_INCLUDE_DIRS}
                )
target_link_libraries(${PROJECT_NAME}CUDA PRIVATE ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS} ${CUVID_LIB} ${OpenCV_LIBS})


file(GLOB source_files_tensorlayer
set_target_properties(${PROJECT_NAME}CUDA PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON
                            CUDA_RUNTIME_LIBRARY Static
                            CUDA_RESOLVE_DEVICE_SYMBOLS ON
                            )

    ./src/*.cpp
    ./src/*.hpp
    ./src/*.h
    ${kernel_tensor_layer_headers}
)
add_executable(${PROJECT_NAME} ${source_files_tensorlayer})
set_property(TARGET ${PROJECT_NAME} PROPERTY
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set_target_properties(${PROJECT_NAME} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RUNTIME_LIBRARY Static
                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                )
target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-lineinfo;--source-in-ptx;>")


target_include_directories(${PROJECT_NAME} PUBLIC
    ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries( ${PROJECT_NAME} ${PROJECT_NAME}CUDA ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS} ${CUVID_LIB} ${OpenCV_LIBS})

file(COPY "data" DESTINATION ${CMAKE_BINARY_DIR})