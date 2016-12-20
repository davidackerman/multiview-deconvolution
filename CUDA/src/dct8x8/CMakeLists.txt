file(GLOB SRC *.cu *.cuh)

set(tgt DCT8x8_CUDA_lib)
cuda_add_library(${tgt} STATIC ${SRC})
target_link_libraries(${tgt} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY} )
set_target_properties(${tgt} PROPERTIES LINKER_LANGUAGE CXX)

set(tgt DCT8x8_CUDA_test)
add_executable(${tgt} DCT8x8_Gold.cpp DCT8x8_Gold.h BmpUtil.cpp BmpUtil.h)
target_link_libraries(${tgt} DCT8x8_CUDA_lib)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)
