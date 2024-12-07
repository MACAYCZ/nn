cmake_minimum_required(VERSION 3.29)
project(nn LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})
set(CMAKE_CUDA_EXTENSIONS ${CMAKE_CXX_EXTENSIONS})

list(APPEND NVCC_OPTIONS -Xptxas=-v)

add_library(nn SHARED nn.cu)
set_property(TARGET nn PROPERTY CUDA_ARCHITECTURES 86-real 86-virtual)
target_compile_options(nn PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_OPTIONS}>)

add_executable(test test.cu)
set_property(TARGET test PROPERTY CUDA_ARCHITECTURES 86-real 86-virtual)
target_compile_options(test PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_OPTIONS}>)
target_link_libraries(test PRIVATE nn)
