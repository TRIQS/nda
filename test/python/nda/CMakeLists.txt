# runs a python test
# Example: add_python_test(my_script)
#   where my_script.py is the script
macro(add_python_test testname)

 set(testname ${testname})
 add_test(py_${testname} ${PYTHON_INTERPRETER} ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.py)
 set_property(TEST py_${testname} PROPERTY ENVIRONMENT PYTHONPATH=${CMAKE_SOURCE_DIR}/python/nda_py:${CMAKE_BINARY_DIR}/python/nda_py:./:$ENV{PYTHONPATH})
 #set_property(TEST py_${testname} PROPERTY ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}/python/h5:./:$ENV{PYTHONPATH})


 if(SANITIZER_RT_PRELOAD)
   set_property(TEST py_${testname} APPEND PROPERTY ENVIRONMENT ${SANITIZER_RT_PRELOAD})
 endif()

endmacro(add_python_test)

#---------------------

# Build the pybind11 module
pybind11_add_module(nda_py_converter_test nda_py_converter_test.cpp )
target_link_libraries(nda_py_converter_test PRIVATE pybind11 nda_py python_and_numpy)

# Build the pybind11 module
pybind11_add_module(nda_py_a nda_py_a.cpp )
target_link_libraries(nda_py_a PRIVATE pybind11 nda_py python_and_numpy)

add_python_test(nda_test_converters)

