
cmake_minimum_required ( VERSION 2.8 )

#add flags for the C++11 
if(CMAKE_COMPILER_IS_GNUCXX) 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++11")
endif()

include_directories(../src/)


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



#test units 
IF(NOT TARGET test_load_multiview_img )
	add_executable(test_load_multiview_img test_load_multiview_images.cpp)
	target_link_libraries(test_load_multiview_img multiview_deconvolution_GPU klblib)
ENDIF()


IF(NOT TARGET test_multiview_deconvolution_LR )
	add_executable(test_multiview_deconvolution_LR test_multiview_deconvolution_LR.cpp)
	target_link_libraries(test_multiview_deconvolution_LR multiview_deconvolution_GPU klblib)
ENDIF()

IF(NOT TARGET test_multiview_deconvolution_LR_multiGPU )
	add_executable(test_multiview_deconvolution_LR_multiGPU test_multiview_deconvolution_LR_multiGPU.cpp)
	target_link_libraries(test_multiview_deconvolution_LR_multiGPU multiview_deconvolution_GPU klblib)
ENDIF()

IF(NOT TARGET test_GPU_elementwise_op )
	CUDA_COMPILE(CUDA_CUDA_OBJ STATIC test_gpu_elementwiseOp.cu  OPTIONS ${CUOPTS})
	add_executable(test_GPU_elementwise_op test_gpu_elementwiseOp.cu ${CUDA_CUDA_OBJ})
	target_link_libraries(test_GPU_elementwise_op multiview_deconvolution_GPU klblib)
	TARGET_LINK_LIBRARIES(test_GPU_elementwise_op ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY})
ENDIF()


IF(NOT TARGET test_GPU_convolution_pr )
	CUDA_COMPILE(CUDA_CUDA_OBJ STATIC test_GPU_convolution.cu  OPTIONS ${CUOPTS})
	add_executable(test_GPU_convolution_pr test_GPU_convolution.cu ${CUDA_CUDA_OBJ})
	target_link_libraries(test_GPU_convolution_pr multiview_deconvolution_GPU klblib)
	TARGET_LINK_LIBRARIES(test_GPU_convolution_pr ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY})
ENDIF()

IF(NOT TARGET test_GPU_TV )	
	add_executable(test_GPU_TV test_total_variation.cxx )
	target_link_libraries(test_GPU_TV multiview_deconvolution_GPU klblib)
	TARGET_LINK_LIBRARIES(test_GPU_TV ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY})
ENDIF()

