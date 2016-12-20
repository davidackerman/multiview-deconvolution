file(GLOB SRC *.c *.h)

set(tgt imWarpFastLib)
add_library(${tgt} STATIC ${SRC})
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})

#tests
set(tgt affine_3d_ops_test)
add_executable(${tgt} test_affine_3d_ops.cxx)
target_link_libraries(${tgt} imWarpFastLib)
add_test(test-${tgt} ${tgt})
set_target_properties(${tgt} PROPERTIES FOLDER test)
