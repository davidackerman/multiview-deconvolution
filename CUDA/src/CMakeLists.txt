#
# CUDA
#
SET(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20;-gencode arch=compute_20,code=sm_21;-gencode arch=compute_30,code=sm_30;-gencode arch=compute_32,code=sm_32;-gencode arch=compute_35,code=sm_35;-gencode arch=compute_50,code=sm_50" CACHE STRING "Semi-colon delimit multiple arguments")
SET(CUDA_NVCC_FLAGS_DEBUG -g; -G;-O0 CACHE STRING "Semi-colon delimit multiple arguments") #set before FIND_PACKAGE(CUDA) in order to avoid FORCE to show them in GUI. So user can modify them
SET(CUDA_NVCC_FLAGS_RELEASE -O3 CACHE STRING "Semi-colon delimit multiple arguments")
SET(CUDA_NVCC_FLAGS_RELWITHDEBINFO -O3 CACHE STRING "Semi-colon delimit multiple arguments")

FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD ON FORCE)

file(GLOB CUDA_SHARED_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/bin/*.dll")
install(FILES ${CUDA_SHARED_LIBS} DESTINATION bin)

#
#submodules
#
add_subdirectory(external/klb/src klb)
add_subdirectory(separableConvolution3D_texture)
add_subdirectory(dct8x8)
add_subdirectory(external/imWarpFast imWarpFast)
add_subdirectory(external/imwarp_flexible imwarp_flexible)

set_target_properties(
	klb klb_static
	DCT8x8_CUDA_lib
	imWarpFastLib
	imwarp_flexible_lib
	SeparableConvolution3D_CUDA_lib
	PROPERTIES FOLDER external)

#
# submodules: add source and header files from subfolders
#
file(GLOB CUDA_HDRS *.h)
file(GLOB CUDA_CUDA_SRCS *.cu)
file(GLOB CPP_SRCS *.cpp)
file(GLOB External_xmlParser_SRCS external/xmlParser/*.cpp)
file(GLOB External_xmlParser_HDRS external/xmlParser/*.h)

set(tgt multiview_deconvolution_GPU)
cuda_add_library(${tgt} STATIC
	${CPP_SRCS} ${CUDA_SRCS} ${CUDA_HDRS} ${CUDA_CUDA_SRCS}
	${External_xmlParser_SRCS} ${External_xmlParser_HDRS})
target_link_libraries(${tgt}
	SeparableConvolution3D_CUDA_lib
	DCT8x8_CUDA_lib
	imWarpFastLib
	imwarp_flexible_lib
	klb_static)
#install(TARGETS ${tgt} DESTINATION lib)

set(tgt multiview_deconvolution_LR_multiGPU)
add_executable(${tgt} main_multiviewDeconvLR_multiGPU.cxx )
target_link_libraries(${tgt} multiview_deconvolution_GPU)
install(TARGETS ${tgt} DESTINATION bin)

#set(tgt multiview_deconvolution_LR_multiGPU_low_mem)
#add_executable(${tgt} main_multiviewDeconvLR_multiGPU_low_mem.cxx )
#target_link_libraries(${tgt} multiview_deconvolution_GPU)
#install(TARGETS ${tgt} DESTINATION bin)

set(tgt main_multiviewDeconvLR_multiGPU_blocksZ)
add_executable(${tgt} main_multiviewDeconvLR_multiGPU_blocksZ.cxx )
target_link_libraries(${tgt} multiview_deconvolution_GPU)
install(TARGETS ${tgt} DESTINATION bin)

set(tgt mvd_without_mvd)
add_executable(${tgt} mvd_without_mvd.cxx )
target_link_libraries(${tgt} multiview_deconvolution_GPU)
install(TARGETS ${tgt} DESTINATION bin)

set(tgt run_trim_psf)
add_executable(${tgt} run_trim_psf.cxx )
target_link_libraries(${tgt} multiview_deconvolution_GPU)
install(TARGETS ${tgt} DESTINATION bin)

set(tgt run_transform_lattice)
add_executable(${tgt} run_transform_lattice.cxx )
target_link_libraries(${tgt} multiview_deconvolution_GPU)
install(TARGETS ${tgt} DESTINATION bin)

set(tgt run_imwarp_flexible)
add_executable(${tgt} run_imwarp_flexible.cxx )
target_link_libraries(${tgt} multiview_deconvolution_GPU)
install(TARGETS ${tgt} DESTINATION bin)

set(tgt run_imWarpFast)
add_executable(${tgt} run_imWarpFast.cxx )
target_link_libraries(${tgt} multiview_deconvolution_GPU)
install(TARGETS ${tgt} DESTINATION bin)
