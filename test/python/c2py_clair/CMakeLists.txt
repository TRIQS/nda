# fetch c2py
include(FetchContent)
FetchContent_Declare(
  c2py
  GIT_REPOSITORY  https://github.com/flatironinstitute/c2py
  GIT_TAG         unstable
  EXCLUDE_FROM_ALL # exclude from all so that c2py is NOT installed.
)
FetchContent_MakeAvailable(c2py)
include(${c2py_SOURCE_DIR}/share/cmake/clair_c2py_generate_bindings.cmake)

# find clair-c2py
find_program(clair-c2py clair-c2py REQUIRED)
message(STATUS "clair-c2py found at ${clair-c2py}")

# Build and install any python modules
file(GLOB_RECURSE module_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
foreach(module_src ${module_sources})
  get_filename_component(module_name ${module_src} NAME_WE)

  Python_add_library(${module_name} MODULE ${module_src})
  target_link_libraries(${module_name} PRIVATE c2py::c2py nda::nda_c)

  clair_c2py_generate_bindings(${module_name})
endforeach()

# Add tests
find_package(Python COMPONENTS Interpreter)
set(all_tests test_nda_converter)
foreach(test_name ${all_tests})
  add_test(NAME Py_${test_name} COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT PYTHONPATH=${PROJECT_BINARY_DIR}/python:${h5_BINARY_DIR}/python:${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH} ${SANITIZER_RT_PRELOAD})
endforeach()
