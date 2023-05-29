cmake_minimum_required (VERSION 3.0)
project (raytrace)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

set (CMAKE_CXX_FLAGS "-fopenmp")
add_compile_options (-Wall -Wextra -Werror=pedantic -Ofast)
add_executable (main 3dendzsinke.cpp framework.cpp)

target_link_libraries (main GL glut GLEW)
