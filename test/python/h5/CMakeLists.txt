# runs a python test
# Example: add_python_test(my_script)
#   where my_script.py is the script
macro(add_python_test testname)

 set(testname ${testname})
 add_test(py_${testname} ${PYTHON_INTERPRETER} ${CMAKE_CURRENT_SOURCE_DIR}/${testname}.py)
 set_property(TEST py_${testname} PROPERTY ENVIRONMENT PYTHONPATH=${CMAKE_SOURCE_DIR}/python/h5:${CMAKE_BINARY_DIR}/python/h5:./:$ENV{PYTHONPATH})
 #set_property(TEST py_${testname} PROPERTY ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}/python/h5:./:$ENV{PYTHONPATH})


 if(SANITIZER_RT_PRELOAD)
   set_property(TEST py_${testname} APPEND PROPERTY ENVIRONMENT ${SANITIZER_RT_PRELOAD})
 endif()

endmacro(add_python_test)

#---------------------

add_python_test(h5_interface)
add_python_test(hdf5_io)

