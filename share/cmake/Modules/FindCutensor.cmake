#  Copyright Simons Foundation 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Locates the cuTENSOR library.
#
# Use -DCutensor_ROOT=/path/to/cutensor (or the environment variable of the
# same name) to point at the unpacked archive. The legacy CUTENSOR_ROOT
# spelling is honoured as well. System paths are searched as a fallback.
#
# Variables set:
#   Cutensor_INCLUDE_DIR, Cutensor_LIBRARIES, Cutensor_FOUND
#
# Target:
#   cutensor (INTERFACE) -- link this. When the resolved library is the static
#   archive, its own dependencies are linked in as well.

if(Cutensor_INCLUDE_DIR AND Cutensor_LIBRARIES)
  set(Cutensor_FIND_QUIETLY TRUE)
endif()

option(Cutensor_USE_STATIC "Prefer libcutensor_static.a over libcutensor.so" OFF)
if(Cutensor_USE_STATIC)
  set(_cutensor_names cutensor_static cutensor)
else()
  set(_cutensor_names cutensor cutensor_static)
endif()

# cuTENSOR header layouts:
#   <root>/include/cutensor.h                                    (tarball)
#   <root>/include/libcutensor/<cuda-major>/cutensor.h           (Debian/Ubuntu apt)
set(_cutensor_incdir_suffixes include)
if(DEFINED CUDAToolkit_VERSION_MAJOR)
  list(PREPEND _cutensor_incdir_suffixes include/libcutensor/${CUDAToolkit_VERSION_MAJOR})
endif()

# The legacy CUTENSOR_ROOT spelling needs explicit hints: find_package only
# picks up Cutensor_ROOT, and it does so before this module is entered.
find_path(Cutensor_INCLUDE_DIR
  NAMES cutensor.h
  HINTS ${CUTENSOR_ROOT} ENV CUTENSOR_ROOT
  PATH_SUFFIXES ${_cutensor_incdir_suffixes}
  DOC "Include directory for cuTENSOR"
)

# cuTENSOR library layouts:
#   <root>/lib/{libcutensor.so,libcutensor_static.a}                   (tarball)
#   <root>/lib/<cuda-major>/{libcutensor.so,...}                       (tarball <= 2.2 only,
#     superseded by one archive per CUDA major)
#   /usr/lib/<arch>/libcutensor/<cuda-major>/{libcutensor.so,...}      (Debian/Ubuntu apt)
set(_cutensor_libdir_suffixes lib)
if(DEFINED CUDAToolkit_VERSION_MAJOR)
  list(PREPEND _cutensor_libdir_suffixes
    lib/${CUDAToolkit_VERSION_MAJOR}
    libcutensor/${CUDAToolkit_VERSION_MAJOR}
  )
endif()

find_library(Cutensor_LIBRARIES
  NAMES ${_cutensor_names}
  HINTS ${Cutensor_INCLUDE_DIR}/.. ${CUTENSOR_ROOT} ENV CUTENSOR_ROOT
  PATH_SUFFIXES ${_cutensor_libdir_suffixes}
  DOC "cuTENSOR library"
)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Cutensor DEFAULT_MSG Cutensor_LIBRARIES Cutensor_INCLUDE_DIR)

mark_as_advanced(Cutensor_INCLUDE_DIR Cutensor_LIBRARIES)

# Interface target -- we refrain from creating an imported target since those
# cannot be exported (see FindMagma.cmake).
if(Cutensor_FOUND AND NOT TARGET cutensor)
  add_library(cutensor INTERFACE)
  target_link_libraries(cutensor INTERFACE ${Cutensor_LIBRARIES})
  target_include_directories(cutensor SYSTEM INTERFACE ${Cutensor_INCLUDE_DIR})

  if(Cutensor_LIBRARIES MATCHES "_static")
    find_package(Threads REQUIRED)
    target_link_libraries(cutensor INTERFACE
      CUDA::cublasLt
      CUDA::cublas
      CUDA::culibos
      Threads::Threads
      ${CMAKE_DL_LIBS}
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_link_libraries(cutensor INTERFACE rt)
    endif()
  endif()
endif()
