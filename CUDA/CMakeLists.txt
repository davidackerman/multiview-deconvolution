project ( multiview_deconvolution )
cmake_minimum_required ( VERSION 2.8 )

#folder where all the executables go (this is seens by all elements called using add_subdirectory)
SET(PROJECT_BINARY_FOLDER "${PROJECT_SOURCE_DIR}/bin")


#contains main source files
add_subdirectory("${PROJECT_SOURCE_DIR}/src")

#contains files to test different aspects of the code
add_subdirectory("${PROJECT_SOURCE_DIR}/test")


#copy binaries to specific folder for release version
#INSTALL(TARGETS  klblib klblibs test_klbIO test_klbIO_C DESTINATION "${KLB_BINARY_FOLDER}" CONFIGURATIONS Release)
