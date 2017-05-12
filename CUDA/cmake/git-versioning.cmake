FIND_PACKAGE(git QUIET)
if(NOT GIT_FOUND)
    set(GIT_EXECUTABLE git)
endif()

exec_program(${GIT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR} ARGS "describe --tags" OUTPUT_VARIABLE GIT_TAG)
exec_program(${GIT_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR} ARGS "describe --always" OUTPUT_VARIABLE GIT_HASH)
add_definitions(-DGIT_TAG=${GIT_TAG} -DGIT_HASH=${GIT_HASH})
set(CPACK_PACKAGE_VERSION ${GIT_TAG})
message("Version ${GIT_TAG} ${GIT_HASH}")
