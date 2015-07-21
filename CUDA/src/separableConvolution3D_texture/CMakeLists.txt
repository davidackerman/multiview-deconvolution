#CMake file to build Tracking_GMM project

#define important variables for the rest of the project
set(SETBYUSER_CUDA_ARCH sm_20 CACHE STRING "CUDA architecture") #for Quadro FX4800 sm13;fir Tesla sm_20


#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (SeparableConvolution3D_CUDA_project)



#
#Set important flags
#

# Call  "cmake-gui -DCMAKE_BUILD_TYPE=Release .." on the command line to specify a particular build, take their definition
# and dump it in the cache along with proper documentation, otherwise set CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
#
IF( CMAKE_BUILD_TYPE STREQUAL "")
	SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ELSE() #Debug is the default
	SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

MESSAGE("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

#to locate scripts
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake") 

#
# Find necessary libraries
#
#Cuda
SET(CUDA_NVCC_FLAGS_DEBUG -g; -G;-O0;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments") #set before FIND_PACKAGE(CUDA) in order to avoid FORCE to show them in GUI. So user can modify them
SET(CUDA_NVCC_FLAGS_RELEASE -O3;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments")
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD ON FORCE)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}) 
#lcudart and lcuda  are already added as variables with find package
#SET(LIBS_CUDA_CUSPARSE "cusparse" CACHE STRING "")#add CUDA libraries ignored in CUDA.cmake


INCLUDE_DIRECTORIES (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")


#
# submodules: add source and header files from subfolders
#
file(GLOB CUDA_HDRS *.h)
file(GLOB CUDA_CUDA_SRCS *.cu)



CUDA_COMPILE(CUDA_CUDA_OBJ STATIC ${CUDA_CUDA_SRCS} OPTIONS ${CUOPTS})

#compile executable
IF(NOT TARGET SeparableConvolution3D_CUDA_test)

add_executable(SeparableConvolution3D_CUDA_test mainTest.cxx convolutionTexture_gold.cxx ${CUDA_SRCS} ${CUDA_CUDA_OBJ} ${CUDA_HDRS} ${CUDA_CUDA_SRCS} )
target_link_libraries(SeparableConvolution3D_CUDA_test ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY} )

ENDIF()



#generate also library
IF(NOT TARGET SeparableConvolution3D_CUDA_lib)
add_library(SeparableConvolution3D_CUDA_lib ${CUDA_SRCS} ${CUDA_CUDA_OBJ} ${CUDA_HDRS} ${CUDA_CUDA_SRCS})
#link libraries
TARGET_LINK_LIBRARIES(SeparableConvolution3D_CUDA_lib ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY} )
SET_TARGET_PROPERTIES(SeparableConvolution3D_CUDA_lib PROPERTIES LINKER_LANGUAGE C)
ENDIF()