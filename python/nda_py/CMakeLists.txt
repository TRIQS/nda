# Configure the version
configure_file(version.py.in version.py)

# nda_py library
add_library(nda_py nda_py.cpp)

# Enable warnings
target_link_libraries(nda_py PRIVATE nda_c python_and_numpy $<BUILD_INTERFACE:nda_warnings>)

# Configure target and compilation
target_compile_options(nda_py PUBLIC -fPIC -std=c++17 $<$<CXX_COMPILER_ID:GNU>:-fconcepts>)
target_include_directories(nda_py PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/python> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)
target_include_directories(nda_py SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_compile_definitions(nda_py PUBLIC
				NDA_GIT_HASH=${PROJECT_GIT_HASH}
				$<$<CONFIG:Debug>:NDA_DEBUG>
			  )



# Install library and headers
install(TARGETS nda_py python_and_numpy EXPORT nda-targets DESTINATION lib)
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
    set_target_properties(nda_py PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  else()
    message(STATUS "clang-tidy not found in $PATH. Please consider installing clang-tidy for additional checks!")
  endif()

  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    message(STATUS "cppcheck found: ${CPPCHECK_EXECUTABLE}")
    add_custom_command(
	TARGET nda_py
	COMMAND ${CPPCHECK_EXECUTABLE}
	  --enable=warning,style,performance,portability
	  --std=c++17
	  --template=gcc
	  --verbose
	  --force
	  --quiet
          ${sources}
	WORKING_DIRECTORY
	  ${CMAKE_CURRENT_SOURCE_DIR}
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
  if(NOT TARGET asan)
    find_package(sanitizer REQUIRED "asan")
  endif()
  target_link_libraries(nda_py PUBLIC $<BUILD_INTERFACE:asan>)
endif()
if(UBSAN)
  if(NOT TARGET ubsan)
    find_package(sanitizer REQUIRED "ubsan")
  endif()
  target_link_libraries(nda_py PUBLIC $<BUILD_INTERFACE:ubsan>)
endif()

if(MSAN)
  if(NOT TARGET msan)
    add_library(msan INTERFACE)
    target_compile_options(msan INTERFACE -fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=all -fsanitize-memory-track-origins=2 -O1 -ggdb3)
    target_link_libraries(msan INTERFACE "-fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=all -fsanitize-memory-track-origins=2")
  endif()
  target_link_libraries(nda_py PUBLIC $<BUILD_INTERFACE:msan>)
endif()
