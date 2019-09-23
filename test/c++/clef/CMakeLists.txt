# List of all tests
file(GLOB_RECURSE all_tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_executable(${test_name} ${test})
  set_property(TARGET ${test_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  target_link_libraries(${test_name} clef_c gtest_main project_warnings)

  # ADDED OP
  target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/c++)
  target_compile_options(${test_name} PUBLIC -fPIC -std=c++17)

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

