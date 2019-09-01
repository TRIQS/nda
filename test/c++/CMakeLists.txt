# Copy reference h5 files to binary dir
file(GLOB all_h5_files *.h5)
file(COPY ${all_h5_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# The list of all c++ tests
set(all_tests nda_idx_map nda_storage nda_basic nda_expr_template nda_map)

#=== build gtest as an external project ===

include(ExternalProject)
set(GTEST_ROOT ${CMAKE_BINARY_DIR}/gtest)
find_package(gtest)
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

#=== Add all tests and register them for the static analyzer checks ===

foreach(t ${all_tests})
  #string(REPLACE ".cpp" "" t ${t1})  

  add_executable(${t} ${CMAKE_CURRENT_SOURCE_DIR}/${t}.cpp)
  target_link_libraries(${t} nda_c gtest)
  add_test(${t} ${CMAKE_CURRENT_BINARY_DIR}/${t})
  # Run clang-tidy if found
  if(CLANG_TIDY_EXECUTABLE)
    set_target_properties(${t} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  endif()
  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    add_custom_command(
      TARGET ${t}
      COMMAND ${CPPCHECK_EXECUTABLE}
      --enable=warning,style,performance,portability
      --std=c++14
      --template=gcc
      --verbose
      --quiet
      ${CMAKE_CURRENT_SOURCE_DIR}/${t}.cpp
    )
  endif()
endforeach()

# Set UBSan Options for all tests
set_property(TEST ${all_tests} PROPERTY
  ENVIRONMENT
    UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1:$ENV{UBSAN_OPTIONS}
)
