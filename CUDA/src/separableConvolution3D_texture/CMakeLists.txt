file(GLOB SRCS *.h *.cu)

set(tgt SeparableConvolution3D_CUDA_lib)
cuda_add_library(${tgt} ${CUDA_SRCS} ${SRCS})
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(${tgt} PROPERTIES LINKER_LANGUAGE CXX)

set(tgt SeparableConvolution3D_CUDA_test)
add_executable(${tgt} mainTest.cxx convolutionTexture_gold.cxx)
target_link_libraries(${tgt} SeparableConvolution3D_CUDA_lib)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)
