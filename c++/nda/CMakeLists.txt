file(GLOB_RECURSE sources *.cpp)
if(NOT CudaSupport)
  list(FILTER sources EXCLUDE REGEX "(cublas_interface|cusolver_interface|cutensor_interface)")
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
if(LAPACK_LIBRARIES MATCHES "mkl")
  target_compile_definitions(${PROJECT_NAME}_c PUBLIC NDA_HAVE_GEMM_BATCH)
endif()

# Link against interface target and export
target_link_libraries(${PROJECT_NAME}_c PRIVATE blas_lapack)
install(TARGETS blas_lapack EXPORT ${PROJECT_NAME}-targets)

# ========= CudaRT, CuBLAS, CuSolver & CuTensor ==========

if(CudaSupport)
  message(STATUS "-------- CUDAToolkit detection -------------")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${PROJECT_NAME}_c PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver)
  target_compile_definitions(${PROJECT_NAME}_c PUBLIC NDA_HAVE_CUDA NDA_HAVE_DEVICE)

  message(STATUS "-------- cutensor detection -------------")
  option(Use_Cutensor "Use cutensor library" OFF)
  if(Use_Cutensor)
    if(NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
      message(FATAL_ERROR "CUTENSOR_ROOT not set!")
    endif() 
    if(DEFINED ENV{CUTENSOR_ROOT})
      set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
    endif()
    message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
    if(NOT EXISTS ${CUTENSOR_ROOT})
      message(FATAL_ERROR "Cannot find CUTENSOR_ROOT")
    endif()

    if(NOT TARGET cutensor)
      add_library(cutensor SHARED IMPORTED)
      if(WIN32)
        set(CUTENSOR_LIBRARY_NAME "cutensor.dll")
        set(CUTENSOR_LIBRARY_DEF "cutensor.lib")
      else()
        set(CUTENSOR_LIBRARY_NAME "libcutensor.so")
      endif()
      if((${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11) AND (${CUDAToolkit_VERSION_MINOR} LESS_EQUAL 0))
        set(LIB_DIR "/lib/11.0")
      elseif((${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11) AND (${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 1))
        set(LIB_DIR "/lib/11")
      elseif((${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 12))
        set(LIB_DIR "/lib/12")
      endif()
      set_target_properties(cutensor PROPERTIES
        IMPORTED_LOCATION "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_NAME}"
        IMPORTED_IMPLIB "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_DEF}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")
    endif()
    target_link_libraries(${PROJECT_NAME}_c PUBLIC cutensor)
    target_compile_definitions(${PROJECT_NAME}_c PUBLIC NDA_HAVE_CUTENSOR)
  endif()
endif()

# ========= Magma ==========

if(Use_Magma)
  message(STATUS "-------- Magma detection -------------")
  find_package(Magma REQUIRED)
  target_link_libraries(${PROJECT_NAME}_c PUBLIC magma)
  target_compile_definitions(${PROJECT_NAME}_c PUBLIC NDA_HAVE_MAGMA)
  install(TARGETS magma EXPORT ${PROJECT_NAME}-targets)
endif()

# ========= TBLIS ==========

message(STATUS "-------- TBLIS detection -------------")
option(Use_Tblis "Use TBLIS library" OFF)
if(Use_Tblis)
  # no find package, :-(, doing it by hand
  if(NOT DEFINED TBLIS_ROOT AND DEFINED ENV{TBLIS_ROOT})
    set(TBLIS_ROOT $ENV{TBLIS_ROOT})
  endif()
  if(NOT DEFINED TBLIS_ROOT)
    message(FATAL_ERROR "Error: Please set TBLIS_ROOT when Use_Tblis is set to ON")
  endif()
  message(STATUS "TBLIS_ROOT: ${TBLIS_ROOT}") 
  # MAM: improve this, need to check if library is in TBLIS_ROOT/include, TBLIS_ROOT/lib
  add_library(tblis INTERFACE IMPORTED)
  set_target_properties(tblis PROPERTIES 
        INTERFACE_INCLUDE_DIRECTORIES  "${TBLIS_ROOT}/include"
        INTERFACE_LINK_LIBRARIES "-L${TBLIS_ROOT}/lib;-ltblis;-ltci;-Wl,-rpath=${TBLIS_ROOT}/lib")
  target_link_libraries(${PROJECT_NAME}_c PUBLIC tblis)
  target_compile_definitions(${PROJECT_NAME}_c PUBLIC NDA_HAVE_TBLIS)
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
