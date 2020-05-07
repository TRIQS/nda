# Build the cpp2py module
add_cpp2py_module(nda_py_a)
target_link_libraries(nda_py_a nda_py)
target_include_directories(nda_py_a PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

## Build the pybind11 module
#pybind11_add_module(nda_py_converter_test nda_py_converter_test.cpp )
#target_link_libraries(nda_py_converter_test PRIVATE pybind11 nda_py python_and_numpy)

## Build the pybind11 module
#pybind11_add_module(nda_py_a nda_py_a.cpp)
#target_link_libraries(nda_py_a PRIVATE pybind11 nda_py python_and_numpy)

# List of all tests
set(all_tests nda_test_converters)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_test(NAME Py_${test_name} COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/${test_dir}:${PROJECT_BINARY_DIR}/python:$ENV{PYTHONPATH} ${SANITIZER_RT_PRELOAD})
endforeach()
