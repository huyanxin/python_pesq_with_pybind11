cmake_minimum_required(VERSION 2.8.12)
project(python_pesq)
add_definitions(-Wall -Wwrite-strings)
include_directories(include)
add_subdirectory(pybind11)
#pybind11_add_module(python_pesq src/python_pesq.cpp src/pesqdsp.c src/pesqmod.c src/dsp.c)

add_library(python_pesq MODULE src/python_pesq.cpp src/pesqdsp.cpp src/pesqmod.cpp src/dsp.cpp)
target_link_libraries(python_pesq PRIVATE pybind11::module)
target_link_libraries(python_pesq PRIVATE m)
set_target_properties(python_pesq  PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                                 SUFFIX "${PYTHON_MODULE_EXTENSION}")
