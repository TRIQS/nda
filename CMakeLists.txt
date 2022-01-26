# Copyright (c) 2018-2022 Simons Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Philipp Dumitrescu, Olivier Parcollet, Dylan Simon, Nils Wentzell

cmake_minimum_required(VERSION 3.12.4 FATAL_ERROR)
cmake_policy(VERSION 3.12.4)
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# ############
# Define Project
project(nda VERSION 1.1.0 LANGUAGES C CXX CUDA)
get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)

# Get the git hash & print status
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE PROJECT_GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "${PROJECT_NAME} version : ${PROJECT_VERSION}")
message(STATUS "${PROJECT_NAME} Git hash: ${PROJECT_GIT_HASH}")

# Assert that Install directory is given and invalid.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  message(FATAL_ERROR "No install prefix given (or invalid)")
endif()
if(NOT IS_SUBPROJECT)
  message(STATUS "-------- CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} --------")
endif()
set(${PROJECT_NAME}_BINARY_DIR ${PROJECT_BINARY_DIR} CACHE STRING "Binary directory of the ${PROJECT_NAME} Project")

# Make additional Find Modules available
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/share/cmake/Modules)

# ############
# CMake Options

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
if(NOT IS_SUBPROJECT)
  message(STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} --------")
endif()

# Python Support
option(PythonSupport "Build with Python support" ON)

# Documentation
option(Build_Documentation "Build documentation" OFF)
if(Build_Documentation AND NOT PythonSupport)
  message(FATAL_ERROR "Build_Documentation=ON requires PythonSupport to be enabled")
endif()

# Benchmarks
option(Build_Benchs "Build Benchmarks" OFF)

# Testing
option(Build_Tests "Build tests" ON)
option(Build_SSO_Tests "Build same tests with various SSO/Custom Allocators" OFF)
if(Build_Tests)
  enable_testing()
endif()

# Build static libraries by default
option(BUILD_SHARED_LIBS "Enable compilation of shared libraries" OFF)

# ############
# Global Compilation Settings

# Export the list of compile-commands into compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Disable compiler extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# Provide additional debugging information for Debug builds
add_compile_options($<$<CONFIG:Debug>:-ggdb3>)

# Create an Interface target for compiler warnings
add_library(${PROJECT_NAME}_warnings INTERFACE)
target_compile_options(${PROJECT_NAME}_warnings
  INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-sign-compare
    $<$<CXX_COMPILER_ID:GNU>:-Wno-comma-subscript>
    $<$<CXX_COMPILER_ID:GNU>:-Wshadow=local>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-comma-subscript>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-warning-option>
    $<$<CXX_COMPILER_ID:Clang>:-Wshadow>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-gcc-compat>
    $<$<CXX_COMPILER_ID:Clang>:-Wno-c++20-extensions>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wno-deprecated-comma-subscript>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wno-unknown-warning-option>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wshadow>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wno-gcc-compat>
    $<$<CXX_COMPILER_ID:AppleClang>:-Wno-c++20-extensions>
)

# ---------------------------------
# Resolve Clang Linktime Problems
# CMake will adjust any linker flags from '-L path_to/mylib.so' to -lmylib
# if the proper mylib.so is automatically found by the linker, i.e.
# the directory comes first in LIBRARY_PATH.
# The clang linker however ignores LIBRARY_PATH.
# We thus explicitly add the content of LIBRARY_PATH to the LDFLAGS
# FIXME For future cmake versions we should populate the
# INTERFACE_LINK_DIRECTORIES of the triqs target
# ---------------------------------
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND DEFINED ENV{LIBRARY_PATH})
  string(REPLACE ":" ";" LINK_DIRS $ENV{LIBRARY_PATH})
  foreach(dir ${LINK_DIRS})
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -L${dir}")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS " -L${dir}")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -L${dir}")
  endforeach()
endif()

# #############
# Build Project

enable_testing()
# Find / Build dependencies
add_subdirectory(deps)

# Build and install the library
add_subdirectory(c++/${PROJECT_NAME})

# Tests
if(Build_Tests)
  add_subdirectory(test)
endif()

# Python
if(PythonSupport)
  add_subdirectory(python/nda_py)
endif()

# Benchmarks
if(Build_Benchs AND NOT ASAN AND NOT UBSAN)
  add_subdirectory(benchmarks)
endif()

# Docs
if(Build_Documentation)
  add_subdirectory(doc)
endif()

# Additional configuration files
add_subdirectory(share)

# #############
# Debian Package

option(BUILD_DEBIAN_PACKAGE "Build a deb package" OFF)
if(BUILD_DEBIAN_PACKAGE AND NOT IS_SUBPROJECT)
  if(NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr")
    message(FATAL_ERROR "CMAKE_INSTALL_PREFIX must be /usr for packaging")
  endif()
  set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
  set(CPACK_GENERATOR "DEB")
  set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
  set(CPACK_PACKAGE_CONTACT "https://github.com/TRIQS/${PROJECT_NAME}")
  execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CMAKE_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
  include(CPack)
endif()
