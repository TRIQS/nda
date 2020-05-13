file(GLOB_RECURSE sources *.cpp)
add_library(nda_c ${sources})
add_library(nda::nda_c ALIAS nda_c)

# Enable warnings
target_link_libraries(nda_c PRIVATE $<BUILD_INTERFACE:nda_warnings>)

# Configure compilation
target_compile_options(nda_c PUBLIC -std=c++17 $<$<CXX_COMPILER_ID:GNU>:-fconcepts>)
set_property(TARGET nda_c PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nda_c PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)
target_include_directories(nda_c SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_compile_definitions(nda_c PUBLIC
				NDA_GIT_HASH=${PROJECT_GIT_HASH}
				$<$<CONFIG:Debug>:NDA_DEBUG>
			  )

# Install library and headers
install(TARGETS nda_c EXPORT nda-targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")


# ========= Additional Depdencies ==========

# Link against HDF5 C++ Interface
target_link_libraries(nda_c PUBLIC h5::h5_c)

# Link against MPI C++ Interface
target_link_libraries(nda_c PUBLIC mpi::mpi_c)


# ========= Blas / Lapack ==========

message(STATUS "-------- Lapack detection -------------")

find_package(LAPACK REQUIRED)

# Create an interface target
add_library(blas_lapack INTERFACE)
add_library(nda::blas_lapack ALIAS blas_lapack)
list(REMOVE_DUPLICATES LAPACK_LIBRARIES)
target_link_libraries(blas_lapack INTERFACE ${LAPACK_LIBRARIES})
target_compile_options(blas_lapack INTERFACE ${LAPACK_LINKER_FLAGS})

# Link against interface target and export
target_link_libraries(nda_c PRIVATE blas_lapack)
install(TARGETS blas_lapack EXPORT nda-targets)


# ========= Static Analyzer Checks ==========

option(ANALYZE_SOURCES OFF "Run static analyzer checks if found (clang-tidy, cppcheck)")
if(ANALYZE_SOURCES)

  # Locate static analyzer tools
  find_program(CPPCHECK_EXECUTABLE NAMES "cppcheck" PATHS ENV PATH)
  find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy" PATHS ENV PATH)

  # Run clang-tidy if found
  if(CLANG_TIDY_EXECUTABLE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXECUTABLE}")
    set_target_properties(nda_c PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  else()
    message(STATUS "clang-tidy not found in $PATH. Please consider installing clang-tidy for additional checks!")
  endif()

  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    message(STATUS "cppcheck found: ${CPPCHECK_EXECUTABLE}")
    add_custom_command(
	TARGET nda_c
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
  if(NOT TARGET asan)
    find_package(sanitizer REQUIRED "asan")
  endif()
  target_link_libraries(nda_c PUBLIC $<BUILD_INTERFACE:asan>)
endif()
if(UBSAN)
  if(NOT TARGET ubsan)
    find_package(sanitizer REQUIRED "ubsan")
  endif()
  target_link_libraries(nda_c PUBLIC $<BUILD_INTERFACE:ubsan>)
endif()

if(MSAN)
  if(NOT TARGET msan)
    add_library(msan INTERFACE)
    target_compile_options(msan INTERFACE -fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=all -fsanitize-memory-track-origins=2 -O1 -ggdb3)
    target_link_libraries(msan INTERFACE "-fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=all -fsanitize-memory-track-origins=2")
  endif()
  target_link_libraries(nda_c PUBLIC $<BUILD_INTERFACE:msan>)
endif()
