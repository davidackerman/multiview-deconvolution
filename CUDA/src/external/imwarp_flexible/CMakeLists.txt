#CMake file to build more flexible multi-threaded affine transformation code



cmake_minimum_required (VERSION 2.8)
project (imwarp_flexible_project)
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
# submodules: add source and header files from subfolders
#
file(GLOB C_HDRS *.h)

file(GLOB C_SRCS *.c)


#generate library
IF(NOT TARGET imwarp_flexible_lib )
	add_library(imwarp_flexible_lib ${C_HDRS} ${C_SRCS})
ENDIF()

# #standalone executable that can be used for testing
# IF(NOT TARGET run_imwarp_flexible )
	# add_executable(run_imwarp_flexible run_imwarp_flexible.cpp)
	# target_link_libraries(run_imwarp_flexible imwarp_flexible_lib multiview_deconvolution_GPU)
# ENDIF()
