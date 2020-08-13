add_subdirectory(c++)

if(PythonSupport AND NOT MSAN)
 add_subdirectory(python)
endif()
