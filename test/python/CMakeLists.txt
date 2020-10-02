# Build the cpp2py module
add_cpp2py_module(wrap_basic)
target_link_libraries(wrap_basic nda_py)
target_include_directories(wrap_basic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Build the cpp2py module
add_cpp2py_module(copy_move_stat)
target_link_libraries(copy_move_stat nda_py)
target_include_directories(copy_move_stat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

## Build the pybind11 module
#pybind11_add_module(pybind_a pybind_a.cpp)
#target_link_libraries(pybind_a PRIVATE pybind11 nda_py python_and_numpy)

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
