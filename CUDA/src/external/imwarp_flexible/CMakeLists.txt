set(tgt imwarp_flexible_lib)
add_library(${tgt} STATIC imwarp_flexible.c image_interpolation_ml.c)
#message("The CMAKE_CURRENT_LIST_DIR is ${CMAKE_CURRENT_LIST_DIR}")
target_include_directories(${tgt} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
