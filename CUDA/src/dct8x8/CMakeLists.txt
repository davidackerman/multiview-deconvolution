#CMake file to for 8x8 DCT based on NVidia CUDA samples


#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (DCT8x8_CUDA_project)



#
#Set important flags
#


MESSAGE("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

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

FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD ON FORCE)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}) 
#lcudart and lcuda  are already added as variables with find package
#SET(LIBS_CUDA_CUSPARSE "cusparse" CACHE STRING "")#add CUDA libraries ignored in CUDA.cmake


INCLUDE_DIRECTORIES (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")


#
# submodules: add source and header files from subfolders
#
file(GLOB CUDA_HDRS *.cuh)
file(GLOB CUDA_CUDA_SRCS *.cu)



CUDA_COMPILE(CUDA_CUDA_OBJ STATIC ${CUDA_CUDA_SRCS} OPTIONS ${CUOPTS})

#compile executable
IF(NOT TARGET DCT8x8_CUDA_test)

add_executable(DCT8x8_CUDA_test DCT8x8_Gold.cpp DCT8x8_Gold.h BmpUtil.cpp BmpUtil.h ${CUDA_SRCS} ${CUDA_CUDA_OBJ} ${CUDA_HDRS} ${CUDA_CUDA_SRCS} )
target_link_libraries(DCT8x8_CUDA_test ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY} )

ENDIF()



#generate also library
IF(NOT TARGET DCT8x8_CUDA_lib)
	add_library(DCT8x8_CUDA_lib ${CUDA_SRCS} ${CUDA_CUDA_OBJ} ${CUDA_HDRS} ${CUDA_CUDA_SRCS})
	#link libraries
	TARGET_LINK_LIBRARIES(DCT8x8_CUDA_lib ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY} )
	SET_TARGET_PROPERTIES(DCT8x8_CUDA_lib PROPERTIES LINKER_LANGUAGE C)
ENDIF()