
cmake_minimum_required ( VERSION 2.8 )

#add flags for the C++11 
if(CMAKE_COMPILER_IS_GNUCXX) 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++11")
endif()

include_directories(../src/)


IF(NOT TARGET test_load_multiview_img )
	add_executable(test_load_multiview_img test_load_multiview_images.cpp)
	target_link_libraries(test_load_multiview_img TGMM_parser_lib)
ENDIF()


IF(NOT TARGET test_multiview_deconvolution_LR )
	add_executable(test_multiview_deconvolution_LR test_multiview_deconvolution_LR.cpp)
	target_link_libraries(test_multiview_deconvolution_LR TGMM_parser_lib)
ENDIF()

