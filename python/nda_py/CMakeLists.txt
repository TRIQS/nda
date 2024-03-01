# Configure the version
configure_file(version.py.in version.py)

# nda_py interface library
add_library(nda_py INTERFACE)
add_library(${PROJECT_NAME}::nda_py ALIAS nda_py)

# Configure target and compilation
target_compile_features(nda_py INTERFACE cxx_std_20)
target_include_directories(nda_py INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/python> $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)
target_include_directories(nda_py SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_compile_definitions(nda_py INTERFACE
				NDA_GIT_HASH=${PROJECT_GIT_HASH}
				$<$<CONFIG:Debug>:NDA_DEBUG>
			  )



# Install library and headers
install(TARGETS nda_py EXPORT ${PROJECT_NAME}-targets DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
set(PYTHON_LIB_DEST ${CPP2PY_PYTHON_LIB_DEST_ROOT}/nda_py)
install(FILES __init__.py DESTINATION ${PYTHON_LIB_DEST})

# ========= Additional Depdencies ==========

# Link against nda_c
target_link_libraries(nda_py INTERFACE nda::nda_c)

# Link against Cpp2Py
target_link_libraries(nda_py INTERFACE cpp2py::cpp2py)

# Link against Python and Numpy
target_link_libraries(nda_py INTERFACE cpp2py::python_and_numpy)

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
	  --std=c++20
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
option(MSAN OFF "Compile library and executables with LLVM Memory Sanitizer")

if(ASAN)
  if(NOT TARGET asan)
    find_package(sanitizer REQUIRED "asan")
  endif()
  target_link_libraries(nda_py INTERFACE $<BUILD_INTERFACE:asan>)
endif()
if(UBSAN)
  if(NOT TARGET ubsan)
    find_package(sanitizer REQUIRED "ubsan")
  endif()
  target_link_libraries(nda_py INTERFACE $<BUILD_INTERFACE:ubsan>)
endif()

if(MSAN)
  if(NOT TARGET msan)
    add_library(msan INTERFACE)
    target_compile_options(msan INTERFACE -fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=all -fsanitize-memory-track-origins=2 -O1 -ggdb3)
    target_link_libraries(msan INTERFACE "-fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=all -fsanitize-memory-track-origins=2")
  endif()
  target_link_libraries(nda_py INTERFACE $<BUILD_INTERFACE:msan>)
endif()
