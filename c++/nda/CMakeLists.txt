file(GLOB_RECURSE sources *.cpp)
add_library(${PROJECT_NAME}_c ${sources})
add_library(${PROJECT_NAME}::${PROJECT_NAME}_c ALIAS ${PROJECT_NAME}_c)

# Enable warnings
target_link_libraries(${PROJECT_NAME}_c PRIVATE $<BUILD_INTERFACE:${PROJECT_NAME}_warnings>)

# Configure target and compilation
target_compile_features(${PROJECT_NAME}_c PUBLIC cxx_std_20)
set_target_properties(${PROJECT_NAME}_c PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
)
target_include_directories(${PROJECT_NAME}_c PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)
target_include_directories(${PROJECT_NAME}_c SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_compile_definitions(${PROJECT_NAME}_c PUBLIC
				NDA_GIT_HASH=${PROJECT_GIT_HASH}
				$<$<CONFIG:Debug>:NDA_DEBUG>
			  )

# Install library and headers
install(TARGETS ${PROJECT_NAME}_c EXPORT ${PROJECT_NAME}-targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.hxx" PATTERN "*.h")

# Configure and install version file
configure_file(version.hpp.in version.hpp @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.hpp DESTINATION include/${PROJECT_NAME})

# ========= Additional Depdencies ==========

# Link against HDF5 C++ Interface
target_link_libraries(${PROJECT_NAME}_c PUBLIC h5::h5_c)

# Link against MPI C++ Interface
target_link_libraries(${PROJECT_NAME}_c PUBLIC mpi::mpi_c)

# ========= Blas / Lapack ==========

message(STATUS "-------- Lapack detection -------------")

find_package(LAPACK REQUIRED)

# Create an interface target
add_library(blas_lapack INTERFACE)
add_library(${PROJECT_NAME}::blas_lapack ALIAS blas_lapack)
list(REMOVE_DUPLICATES LAPACK_LIBRARIES)
target_link_libraries(blas_lapack INTERFACE ${LAPACK_LIBRARIES})
target_compile_options(blas_lapack INTERFACE ${LAPACK_LINKER_FLAGS})

# Link against interface target and export
target_link_libraries(${PROJECT_NAME}_c PRIVATE blas_lapack)
install(TARGETS blas_lapack EXPORT ${PROJECT_NAME}-targets)

# ========= CudaRT & CuBLAS ==========

message(STATUS "-------- CUDAToolkit detection -------------")

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME}_c PUBLIC CUDA::cudart CUDA::cublas)

# ========= Compiler Wrapper ==========

# Get the compiler and linktime flags of nda_c
include(${PROJECT_SOURCE_DIR}/share/cmake/extract_flags.cmake)
extract_flags(nda_c)

# Generate and install compiler wrapper
configure_file(nda++.in nda++ @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nda++ PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE DESTINATION bin)

# ========= Static Analyzer Checks ==========

option(ANALYZE_SOURCES OFF "Run static analyzer checks if found (clang-tidy, cppcheck)")
if(ANALYZE_SOURCES)

  # Locate static analyzer tools
  find_program(CPPCHECK_EXECUTABLE NAMES "cppcheck" PATHS ENV PATH)
  find_program(CLANG_TIDY_EXECUTABLE NAMES "clang-tidy" PATHS ENV PATH)

  # Run clang-tidy if found
  if(CLANG_TIDY_EXECUTABLE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXECUTABLE}")
    set_target_properties(${PROJECT_NAME}_c PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  else()
    message(STATUS "clang-tidy not found in $PATH. Please consider installing clang-tidy for additional checks!")
  endif()

  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    message(STATUS "cppcheck found: ${CPPCHECK_EXECUTABLE}")
    add_custom_command(
	TARGET ${PROJECT_NAME}_c
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
  target_link_libraries(${PROJECT_NAME}_c PUBLIC $<BUILD_INTERFACE:asan>)
endif()
if(UBSAN)
  if(NOT TARGET ubsan)
    find_package(sanitizer REQUIRED "ubsan")
  endif()
  target_link_libraries(${PROJECT_NAME}_c PUBLIC $<BUILD_INTERFACE:ubsan>)
endif()

if(MSAN)
  if(NOT TARGET msan)
    add_library(msan INTERFACE)
    target_compile_options(msan INTERFACE -fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=all -fsanitize-memory-track-origins=2 -O1 -ggdb3)
    target_link_libraries(msan INTERFACE "-fsanitize=memory -fno-omit-frame-pointer -fsanitize-recover=all -fsanitize-memory-track-origins=2")
  endif()
  target_link_libraries(${PROJECT_NAME}_c PUBLIC $<BUILD_INTERFACE:msan>)
endif()
