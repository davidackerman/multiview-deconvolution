#CMake file to build multi-view deconvolution in CUDA

#define important variables for the rest of the project

# refer to the root source directory of the project as ${TGMM_SOURCE_DIR} and
# to the root binary directory of the project as ${TGMM_BINARY_DIR}. 
#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (multi-view_deconvolution_src)
cmake_policy(SET CMP0054 OLD)



#
#Set important flags
#
#add flags for the C++11 
if(CMAKE_COMPILER_IS_GNUCXX) 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
	LINK_DIRECTORIES("/lib64")
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++11")
endif()


#to locate scripts
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake") 

#
# Find necessary libraries
#
#Cuda
SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20;-gencode arch=compute_20,code=sm_21;-gencode arch=compute_30,code=sm_30;-gencode arch=compute_32,code=sm_32;-gencode arch=compute_35,code=sm_35;-gencode arch=compute_50,code=sm_50" CACHE STRING "Semi-colon delimit multiple arguments")
SET(CUDA_NVCC_FLAGS_DEBUG -g; -G;-O0 CACHE STRING "Semi-colon delimit multiple arguments") #set before FIND_PACKAGE(CUDA) in order to avoid FORCE to show them in GUI. So user can modify them
SET(CUDA_NVCC_FLAGS_RELEASE -O3 CACHE STRING "Semi-colon delimit multiple arguments")
SET(CUDA_NVCC_FLAGS_RELWITHDEBINFO -O3 CACHE STRING "Semi-colon delimit multiple arguments")

FIND_PACKAGE(CUDA 6.5 EXACT REQUIRED)
SET(CUDA_VERBOSE_BUILD ON FORCE)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}) 
#lcudart and lcuda  are already added as variables with find package
#SET(LIBS_CUDA_CUSPARSE "cusparse" CACHE STRING "")#add CUDA libraries ignored in CUDA.cmake

#
#submodules: KLB I/O
#
include_directories("${PROJECT_SOURCE_DIR}/external/klb/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/klb/src" "${CMAKE_CURRENT_BINARY_DIR}/KLB")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/external/klb/src")

#
# separable convolution
#
include_directories("${PROJECT_SOURCE_DIR}/separableConvolution3D_texture")
add_subdirectory("${PROJECT_SOURCE_DIR}/separableConvolution3D_texture" "${CMAKE_CURRENT_BINARY_DIR}/separableConv")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/separableConvolution3D_texture")

#
# dct8x8 to calculate weigths
#
include_directories("${PROJECT_SOURCE_DIR}/dct8x8")
add_subdirectory("${PROJECT_SOURCE_DIR}/dct8x8" "${CMAKE_CURRENT_BINARY_DIR}/dct8x8")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/dct8x8")

#
# code to calculate affine transformatiosn of 3D images fast
#
include_directories("${PROJECT_SOURCE_DIR}/external/imWarpFast")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/imWarpFast" "${CMAKE_CURRENT_BINARY_DIR}/imWarpFast")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/external/imWarpFast")

#
# code to calculate affine transformatiosn of 3D images fast, and flexibly
#
include_directories("${PROJECT_SOURCE_DIR}/external/imwarp_flexible")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/imwarp_flexible" "${CMAKE_CURRENT_BINARY_DIR}/imwarp_flexible")
LINK_DIRECTORIES("${PROJECT_SOURCE_DIR}/external/imwarp_flexible")


#
# submodules: add source and header files from subfolders
#
file(GLOB CUDA_HDRS *.h)
file(GLOB CUDA_CUDA_SRCS *.cu)

file(GLOB CPP_SRCS *.cpp)

file(GLOB External_xmlParser_SRCS external/xmlParser/*.cpp)
file(GLOB External_xmlParser_HDRS external/xmlParser/*.h)

CUDA_COMPILE(CUDA_CUDA_OBJ STATIC ${CUDA_CUDA_SRCS} OPTIONS ${CUOPTS})


#generate library
add_library(multiview_deconvolution_GPU ${CPP_SRCS} ${CUDA_SRCS} ${CUDA_CUDA_OBJ} ${CUDA_HDRS} ${CUDA_CUDA_SRCS} ${External_xmlParser_SRCS} ${External_xmlParser_HDRS})

#link libraries
TARGET_LINK_LIBRARIES(multiview_deconvolution_GPU ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY})
TARGET_LINK_LIBRARIES(multiview_deconvolution_GPU SeparableConvolution3D_CUDA_lib DCT8x8_CUDA_lib imwarpFastLib)
TARGET_LINK_LIBRARIES(multiview_deconvolution_GPU klblib)
SET_TARGET_PROPERTIES(multiview_deconvolution_GPU PROPERTIES LINKER_LANGUAGE CXX)


#executable for multiview deconvolution
ADD_EXECUTABLE(multiview_deconvolution_LR_multiGPU main_multiviewDeconvLR_multiGPU.cxx )
TARGET_LINK_LIBRARIES(multiview_deconvolution_LR_multiGPU multiview_deconvolution_GPU)

#executable for multiview deconvolution with low memory computers (using I/O)
ADD_EXECUTABLE(multiview_deconvolution_LR_multiGPU_low_mem main_multiviewDeconvLR_multiGPU_low_mem.cxx )
TARGET_LINK_LIBRARIES(multiview_deconvolution_LR_multiGPU_low_mem multiview_deconvolution_GPU)

#executable for multiview deconvolution with very large datasets. Deconvolution is performed in blocks along z to fit in CPU and GPU memory
ADD_EXECUTABLE(main_multiviewDeconvLR_multiGPU_blocksZ main_multiviewDeconvLR_multiGPU_blocksZ.cxx )
TARGET_LINK_LIBRARIES(main_multiviewDeconvLR_multiGPU_blocksZ multiview_deconvolution_GPU)
