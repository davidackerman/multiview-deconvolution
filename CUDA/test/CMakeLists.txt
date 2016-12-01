
include_directories(../src/)


#
# Find necessary libraries
#

# CUDA
# set before FIND_PACKAGE(CUDA) in order to avoid FORCE to show them in GUI.
# lcudart and lcuda are already added as variables with find package
SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20;-gencode arch=compute_20,code=sm_21;-gencode arch=compute_30,code=sm_30;-gencode arch=compute_32,code=sm_32;-gencode arch=compute_35,code=sm_35;-gencode arch=compute_50,code=sm_50" CACHE STRING "Semi-colon delimit multiple arguments")
SET(CUDA_NVCC_FLAGS_DEBUG -g; -G;-O0 CACHE STRING "Semi-colon delimit multiple arguments") #
SET(CUDA_NVCC_FLAGS_RELEASE -O3 CACHE STRING "Semi-colon delimit multiple arguments")
SET(CUDA_NVCC_FLAGS_RELWITHDEBINFO -O3 CACHE STRING "Semi-colon delimit multiple arguments")

FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD ON FORCE)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
#SET(LIBS_CUDA_CUSPARSE "cusparse" CACHE STRING "")#add CUDA libraries ignored in CUDA.cmake

#test units
set(tgt test_load_multiview_images)
add_executable(${tgt} ${tgt}.cpp)
target_link_libraries(${tgt} multiview_deconvolution_GPU)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)

set(tgt test_multiview_deconvolution_LR)
add_executable(${tgt} ${tgt}.cpp)
target_link_libraries(${tgt} multiview_deconvolution_GPU)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)

set(tgt test_imWarpFast)
add_executable(${tgt} ${tgt}.cxx)
target_link_libraries(${tgt} multiview_deconvolution_GPU)
target_include_directories(${tgt} PRIVATE
	../src/external/klb/src
	../src/external/imWarpFast)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)

set(tgt test_multiview_deconvolution_LR_multiGPU)
add_executable(${tgt} ${tgt}.cpp)
target_link_libraries(${tgt} multiview_deconvolution_GPU)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)

set(tgt test_GPU_elementwise_op)
cuda_add_executable(${tgt} test_gpu_elementwiseOp.cu)
target_link_libraries(${tgt} multiview_deconvolution_GPU)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)

set(tgt test_GPU_reduction_op)
cuda_add_executable(${tgt} test_gpu_reductionOp.cu)
target_link_libraries(${tgt} multiview_deconvolution_GPU)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)

set(tgt test_GPU_convolution_pr)
cuda_add_executable(${tgt} test_GPU_convolution.cu)
target_link_libraries(${tgt} multiview_deconvolution_GPU)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)

set(tgt test_GPU_TV)
add_executable(${tgt} test_total_variation.cxx )
target_link_libraries(${tgt} multiview_deconvolution_GPU)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)

set(tgt test_GPU_LRweightsDCT)
cuda_add_executable(${tgt} test_GPU_LRweightsDCT.cu)
target_link_libraries(${tgt} multiview_deconvolution_GPU)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)
