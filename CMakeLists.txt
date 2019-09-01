# Start configuration
cmake_minimum_required(VERSION 2.8.12)
project(nda C CXX)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message( STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} --------")

# Build static libraries
set(BUILD_SHARED_LIBS OFF)

# Enable compiler warnings for the whole project
add_compile_options(-Wall
		     $<$<CONFIG:Debug>:-Og>
		     $<$<CONFIG:Debug>:-ggdb3>
		     )

# Build and install the nda library
add_subdirectory(nda)

# Build and install the h5 library
add_subdirectory(h5)

# Build and install the nda python module
if(${TRIQS_WITH_PYTHON_SUPPORT})
  add_subdirectory(python/nda)
endif()

# Tests
option(Build_Tests "Tests" ON)
if(Build_Tests)
  enable_testing()
  add_subdirectory(test)
endif()

# Benchmarks
option(Build_Bench "Benchmarks" ON)
if(Build_Bench)
  enable_testing()
  add_subdirectory(benchmarks)
endif()

# Additional configuration files
add_subdirectory(share)

option(Build_Documentation "Build documentation" OFF)
if(Build_Documentation)
  if(NOT TRIQS_WITH_DOCUMENTATION)
    message(WARNING "TRIQS library has not been compiled with its documentation! Cannot build documentation.")
  else()
    message(STATUS "Documentation Build enabled")
    add_subdirectory(doc)
  endif()
endif()

