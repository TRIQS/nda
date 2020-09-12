# Build the cpp2py module
#add_cpp2py_module(wrap_basic)
#target_link_libraries(wrap_basic nda_py)
#target_include_directories(wrap_basic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

## Build the cpp2py module
#add_cpp2py_module(copy_move_stat)
#target_link_libraries(copy_move_stat nda_py)
#target_include_directories(copy_move_stat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

## Build the pybind11 module
pybind11_add_module(wrap_basic wrap_basic.cxx)
target_link_libraries(wrap_basic PRIVATE nda_py) #pybind11 nda_py python_and_numpy)

pybind11_add_module(copy_move_stat copy_move_stat.cxx)
target_link_libraries(copy_move_stat PRIVATE nda_py) #pybind11 nda_py python_and_numpy)

# Generate the C++2Py Module for an hdf5 storable class
#target_link_libraries(storable PRIVATE h5_c hdf5 h5py_c)
#target_compile_options(storable PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>)

## Build the pybind11 module
#pybind11_add_module(pybind_test pybind_test.cpp)
#target_link_libraries(pybind_test PRIVATE pybind11 nda_py python_and_numpy)

# List of all tests
set(all_tests wrap_basic_test copy_move_stat_test)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_test(NAME Py_${test_name} COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/${test_dir}:${PROJECT_BINARY_DIR}/python:${h5_BINARY_DIR}/python:$ENV{PYTHONPATH} ${SANITIZER_RT_PRELOAD})
endforeach()
