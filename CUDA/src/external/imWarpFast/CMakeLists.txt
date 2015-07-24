#CMake file to build multi-thread affine transformation code



cmake_minimum_required (VERSION 2.8)
project (imwarp_fast_project)



#
#Set important flags
#
#add flags for the C++11 
if(CMAKE_COMPILER_IS_GNUCXX) 
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
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
add_library(imwarpFastLib ${C_HDRS} ${C_SRCS})
