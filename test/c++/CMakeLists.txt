# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests
#file(GLOB_RECURSE all_tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
set(all_tests nda_idx_map nda_storage nda_basic nda_expr_template nda_map nda_alloc_poison nda_algorithms nda_arithmetic nda_array_of_non_pod nda_iterators nda_bound_check ) #nda_h5  nda_stl )

#=== build gtest as an external project ===

include(ExternalProject)
set(GTEST_ROOT ${CMAKE_BINARY_DIR}/gtest)
#find_package(gtest)
if (GTEST_FOUND)
 message(STATUS "Found gtest ")
else()
ExternalProject_Add(gtest_project
  PREFIX gtest
  GIT_REPOSITORY https://github.com/abseil/googletest
  GIT_TAG release-1.8.1
  ## In case you cannot clone gtest, comment the two lines
  ## above and point the SOURCE_DIR below to the downloaded gtest sources
  #SOURCE_DIR path_to_gtest_sources
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GTEST_ROOT} -DCMAKE_INSTALL_LIBDIR=lib
)
endif()

# Create the imported gtest target
add_library(gtest STATIC IMPORTED)
add_dependencies(gtest gtest_project)

set_target_properties(gtest PROPERTIES
  IMPORTED_LOCATION ${GTEST_ROOT}/lib/libgtest.a
  INTERFACE_LINK_LIBRARIES pthread
  ## Populating the INTERFACE_INCLUDE_DIRECTORIES won't work
  ## as the directory does not yet exist
  INTERFACE_COMPILE_OPTIONS "-isystem;${GTEST_ROOT}/include"
)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_executable(${test_name} ${test})
  set_property(TARGET ${test_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  target_link_libraries(${test_name} nda_c gtest pthread project_warnings)
  add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  # Run clang-tidy if found
  if(CLANG_TIDY_EXECUTABLE)
    set_target_properties(${test_name} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  endif()
  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    add_custom_command(
      TARGET ${test_name}
      COMMAND ${CPPCHECK_EXECUTABLE}
      --enable=warning,style,performance,portability
      --std=c++14
      --template=gcc
      --verbose
      --quiet
      ${CMAKE_CURRENT_SOURCE_DIR}/${test}.cpp
    )
  endif()
endforeach()
