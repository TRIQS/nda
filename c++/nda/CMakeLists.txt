file(GLOB_RECURSE sources *.cpp)
add_library(nda_c ${sources})

# Link against dependencies
target_link_libraries(nda_c PRIVATE project_warnings)

# Configure compilation
target_compile_options(nda_c PUBLIC -fPIC -std=c++17)
target_include_directories(nda_c PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)
target_compile_definitions(nda_c PUBLIC
				NDA_GIT_HASH=${PROJECT_GIT_HASH}
				$<$<CONFIG:Debug>:NDA_DEBUG>
			  )

# Install library and headers
install(TARGETS nda_c EXPORT nda-targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")


# ---------------------------------
# Lapack
# ---------------------------------

## --------- FIXME _______________

message(STATUS "-------- Lapack detection -------------")

find_package(LAPACK REQUIRED)

message(STATUS "Lapack libraries : ${LAPACK_LIBRARIES}")
message(STATUS "Lapack vendor : ${BLA_VENDOR}")

list(REMOVE_DUPLICATES LAPACK_LIBRARIES)
add_library(blas_lapack INTERFACE)
target_link_libraries(blas_lapack INTERFACE ${LAPACK_LIBRARIES})
target_compile_options(blas_lapack INTERFACE ${LAPACK_LINKER_FLAGS})

target_link_libraries(nda_c PRIVATE blas_lapack)

install(TARGETS blas_lapack EXPORT  nda-targets)

# ---------------------------------
# MPI :FIXME SAME AS TRIQS
# ---------------------------------

message(STATUS "-------- MPI detection -------------")
# We use MPI in C only, not the C++ bindings.
find_package(MPI)

if(NOT MPI_C_FOUND)
 message(FATAL_ERROR "TRIQS requires MPI")
endif()
message(STATUS "MPI C compiler : ${MPI_C_COMPILER}")
message(STATUS "MPI_COMPILE_FLAGS : ${MPI_C_COMPILE_FLAGS} ")
message(STATUS "MPI_C_INCLUDE_PATH : ${MPI_C_INCLUDE_PATH}")
message(STATUS "MPI_C_LIBRARIES : ${MPI_C_LIBRARIES}")
message(STATUS "MPI_CXX_LIBRARIES : ${MPI_CXX_LIBRARIES}")

# FIXME : we should not link to CXX lib, but openmpi uses them by default apparently on Linux

separate_arguments(MPI_C_COMPILE_FLAGS) # Convert to list

add_library(mpi INTERFACE)
target_include_directories(mpi SYSTEM INTERFACE ${MPI_C_INCLUDE_PATH})
target_link_libraries(mpi INTERFACE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_compile_options(mpi INTERFACE ${MPI_C_COMPILE_FLAGS})

target_link_libraries(nda_c PUBLIC mpi)
target_include_directories(nda_c SYSTEM PUBLIC ${MPI_C_INCLUDE_PATH})
install(TARGETS mpi EXPORT nda-targets)

if(NOT MPIEXEC_EXECUTABLE) # Backward compatibility for older FindMPI.cmake
  set(MPIEXEC_EXECUTABLE ${MPIEXEC} CACHE FILENAME "MPI Executable")
endif()
# Compatibility to Open-MPI 3.0.0: check whether MPI executable has option --oversubscribe and add it 
execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ls ${MPIEXEC_POSTFLAGS} RESULT_VARIABLE HAS_NO_OVERSUBSCRIBE OUTPUT_QUIET ERROR_QUIET)
if(NOT HAS_NO_OVERSUBSCRIBE)
  list(APPEND MPIEXEC_PREFLAGS --oversubscribe)
  set(MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS} CACHE STRING "These flags will be directly before the executable that is being run by mpiexec." FORCE)
endif()


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

if(ASAN)
  find_package(sanitizer REQUIRED "asan")
  target_link_libraries(nda_c PUBLIC asan)
  install(TARGETS asan EXPORT nda-targets)
endif()
if(UBSAN)
  find_package(sanitizer REQUIRED "ubsan")
  target_link_libraries(nda_c PUBLIC ubsan)
  install(TARGETS ubsan EXPORT nda-targets)
endif()
