add_library(nda_python nda_py_interface.cpp)

# Enable warnings
target_link_libraries(nda_python PRIVATE nda_c python_and_numpy project_warnings)

# Configure compilation
target_compile_options(nda_python PUBLIC -fPIC -std=c++17 $<$<CXX_COMPILER_ID:GNU>:-fconcepts>)
target_include_directories(nda_python PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/python> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)
target_compile_definitions(nda_python PUBLIC
				NDA_GIT_HASH=${PROJECT_GIT_HASH}
				$<$<CONFIG:Debug>:NDA_DEBUG>
			  )


# Build the pybind11 module
pybind11_add_module(converter_test converter_test.cpp )
target_link_libraries(converter_test PRIVATE nda_python python_and_numpy)


# Install library and headers
install(TARGETS nda_python EXPORT nda_python-targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")


# ========= Static Analyzer Checks ==========

option(ANALYZE_SOURCES OFF "Run static analyzer checks if found (clang-tidy, cppcheck)")
if(ANALYZE_SOURCES)

  # Locate static analyzer tools
  find_program(CPPCHECK_EXECUTABLE NAMES "cppcheck" PATHS ENV PATH)
  find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy" PATHS ENV PATH)

  # Run clang-tidy if found
  if(CLANG_TIDY_EXECUTABLE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXECUTABLE}")
    set_target_properties(nda_python PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  else()
    message(STATUS "clang-tidy not found in $PATH. Please consider installing clang-tidy for additional checks!")
  endif()

  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    message(STATUS "cppcheck found: ${CPPCHECK_EXECUTABLE}")
    add_custom_command(
	TARGET nda_python
	COMMAND ${CPPCHECK_EXECUTABLE}
	  --enable=warning,style,performance,portability
	  --std=c++17
	  --template=gcc
	  --verbose
	  --quiet
          ${sources}
    )
  else()
    message(STATUS "cppcheck not found in $PATH. Please consider installing cppcheck for additional checks!")
  endif()

endif()


# ========= Dynamic Analyzer Checks ==========

option(ASAN OFF "Compile library and executables with LLVM Address Sanitizer")
option(UBSAN OFF "Compile library and executables with LLVM Undefined Behavior Sanitizer")
option(MBSAN OFF "Compile library and executables with LLVM Memory Sanitizer")

if(ASAN)
 #find_package(sanitizer REQUIRED "asan")
  target_link_libraries(nda_python PUBLIC asan)
  install(TARGETS asan EXPORT nda_python-targets)
endif()
if(UBSAN)
 #find_package(sanitizer REQUIRED "ubsan")
  target_link_libraries(nda_python PUBLIC ubsan)
  install(TARGETS ubsan EXPORT nda_python-targets)
endif()

if(MSAN)
  target_compile_options(nda_python PUBLIC -fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=all -fsanitize-memory-track-origins=2 -O1 -ggdb3)
  target_link_libraries(nda_python PUBLIC "-fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=all -fsanitize-memory-track-origins=2")
endif()
