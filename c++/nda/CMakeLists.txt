file(GLOB_RECURSE sources *.cpp)
if(NOT CudaSupport)
  list(FILTER sources EXCLUDE REGEX "(cublas_interface|cusolver_interface)")
endif()
add_library(${PROJECT_NAME}_c ${sources})
add_library(${PROJECT_NAME}::${PROJECT_NAME}_c ALIAS ${PROJECT_NAME}_c)

# Enable warnings
target_link_libraries(${PROJECT_NAME}_c PRIVATE $<BUILD_INTERFACE:${PROJECT_NAME}_warnings>)

# Configure target and compilation
target_compile_features(${PROJECT_NAME}_c PUBLIC cxx_std_23)
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

# OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(${PROJECT_NAME}_c PUBLIC OpenMP::OpenMP_CXX)

# ========= Blas / Lapack ==========

message(STATUS "-------- Lapack detection -------------")

find_package(LAPACK REQUIRED)
if(LAPACK_LIBRARIES MATCHES "libmkl" AND NOT LAPACK_LIBRARIES MATCHES "libmkl_rt")
  message(STATUS "Found MKL: Making sure to use single dynamic library")
  set(BLA_VENDOR Intel10_64_dyn)
  find_package(LAPACK REQUIRED)
endif()
if(LAPACK_LIBRARIES MATCHES "Accelerate.framework")
  message(STATUS "Can't build with Accelerate Framework on OSX, trying to find OpenBLAS instead.")
  set(BLA_VENDOR OpenBLAS) # Accelerate Framework has issue in zdot
  set(LAPACK_ROOT "${LAPACK_ROOT};$ENV{LAPACK_ROOT};/opt/homebrew/opt/openblas;/usr/local/opt/openblas")
  find_package(LAPACK REQUIRED)
endif()

# Create an interface target
add_library(blas_lapack INTERFACE)
add_library(${PROJECT_NAME}::blas_lapack ALIAS blas_lapack)
list(REMOVE_DUPLICATES LAPACK_LIBRARIES)
target_link_libraries(blas_lapack INTERFACE ${LAPACK_LIBRARIES})
target_compile_options(blas_lapack INTERFACE ${LAPACK_LINKER_FLAGS})
if(LAPACK_LIBRARIES MATCHES "libmkl")
  if(DEFINED ENV{MKLROOT})
    target_include_directories(${PROJECT_NAME}_c PUBLIC $ENV{MKLROOT}/include)
  endif()
  target_compile_definitions(${PROJECT_NAME}_c PUBLIC NDA_USE_MKL)
endif()

# Link against interface target and export
target_link_libraries(${PROJECT_NAME}_c PRIVATE blas_lapack)
install(TARGETS blas_lapack EXPORT ${PROJECT_NAME}-targets)

# ========= CudaRT & CuBLAS ==========

if(CudaSupport)
  message(STATUS "-------- CUDAToolkit detection -------------")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${PROJECT_NAME}_c PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)
  target_compile_definitions(${PROJECT_NAME}_c PUBLIC NDA_HAVE_CUDA NDA_HAVE_DEVICE)
endif()

# ========= Magma ==========

if(Use_Magma)
  message(STATUS "-------- Magma detection -------------")
  find_package(Magma REQUIRED)
  target_link_libraries(${PROJECT_NAME}_c PUBLIC magma)
  target_compile_definitions(${PROJECT_NAME}_c PUBLIC NDA_HAVE_MAGMA)
  install(TARGETS magma EXPORT ${PROJECT_NAME}-targets)
endif()

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
	  --std=c++23
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
if(ASAN)
  if(NOT TARGET asan)
    find_package(sanitizer REQUIRED COMPONENTS asan)
  endif()
  target_link_libraries(${PROJECT_NAME}_c PUBLIC $<BUILD_INTERFACE:asan>)
endif()

option(UBSAN OFF "Compile library and executables with LLVM Undefined Behavior Sanitizer")
if(UBSAN)
  if(NOT TARGET ubsan)
    find_package(sanitizer REQUIRED COMPONENTS ubsan)
  endif()
  target_link_libraries(${PROJECT_NAME}_c PUBLIC $<BUILD_INTERFACE:ubsan>)
endif()

option(MSAN OFF "Compile library and executables with LLVM Memory Sanitizer")
if(MSAN)
  if(NOT TARGET msan)
    find_package(sanitizer REQUIRED COMPONENTS msan)
  endif()
  target_link_libraries(${PROJECT_NAME}_c PUBLIC $<BUILD_INTERFACE:msan>)
endif()

option(TSAN OFF "Compile library and executables with LLVM Thread Sanitizer")
if(TSAN)
  if(NOT TARGET tsan)
    find_package(sanitizer REQUIRED COMPONENTS tsan)
  endif()
  target_link_libraries(${PROJECT_NAME}_c PUBLIC $<BUILD_INTERFACE:tsan>)
endif()
