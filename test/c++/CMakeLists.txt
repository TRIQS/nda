# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests
file(GLOB_RECURSE all_tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
option(Use_Cuda "Use CUDA library" OFF)
if(NOT  Use_Cuda)
list(REMOVE_ITEM all_tests "nda_cublas.cpp")
list(REMOVE_ITEM all_tests "nda_culapack.cpp")
list(REMOVE_ITEM all_tests "nda_cuda.cpp")
endif()

macro(SetUpAllTestWithMacroDef extension macrodef)
foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  string(APPEND test_name "${ARGV0}")
  #MESSAGE("${test_name} with option ${ARGV1}")
  get_filename_component(test_dir ${test} DIRECTORY)
  add_executable(${test_name} ${test})
  target_link_libraries(${test_name} ${PROJECT_NAME}_c gtest_main ${PROJECT_NAME}_warnings)
  target_compile_options(${test_name}  PRIVATE "${ARGV1}")
  set_property(TARGET ${test_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  if(test_name MATCHES ".*mpi.*" AND NOT MSAN)
    add_test(NAME ${test_name}_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/${test_dir}/${test_name} ${MPIEXEC_POSTFLAGS} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  endif()
  # Make sure that we properly treat NaN even with IntelLLVM's aggressive default optimizations
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(${test_name} PRIVATE -fhonor-nans)
  endif()
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
      --std=c++20
      --template=gcc
      --verbose
      --force
      --quiet
      ${CMAKE_CURRENT_SOURCE_DIR}/${test}
    )
  endif()
endforeach()
endmacro()

# Regular tests
SetUpAllTestWithMacroDef("" "")

if (Build_SSO_Tests)
  # Rerun all C++ tests with others allocs, using the SS0 with 2 size for handle of the basic_array
  SetUpAllTestWithMacroDef("_SSO_10" "-DNDA_TEST_SSO=10")
  SetUpAllTestWithMacroDef("_SSO_100" "-DNDA_TEST_SSO=100")
  SetUpAllTestWithMacroDef("_MBUCKET_10" "-DNDA_TEST_DEFAULT_ALLOC_MBUCKET=10")
endif()

