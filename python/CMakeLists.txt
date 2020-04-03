if(NOT TARGET cpp2py)
  add_subdirectory(cpp2py)
endif()

## Pybind 11
#find_package(Python)
#add_subdirectory(pybind11)
## Cf https://github.com/pybind/pybind11/issues/1604
#if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
     #target_compile_options(pybind11 INTERFACE -fsized-deallocation)
#endif()

add_subdirectory(h5)
add_subdirectory(nda_py)
