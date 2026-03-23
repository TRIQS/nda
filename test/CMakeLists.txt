add_subdirectory(c++)

if(PythonSupport AND NOT MSAN)
  add_subdirectory(python/cpp2py)
endif()

if(Build_c2py_clair_Tests AND NOT MSAN)
  add_subdirectory(python/c2py_clair)
endif()
