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
# cuTENSOR ships as a tarball with libraries either in <root>/lib/<cuda-major>
# (legacy) or <root>/lib (>=2.0), and headers in <root>/include.#
# 
# Use -DCUTENSOR_ROOT=/path/to/cutensor (or the environment variable of the
# same name) to point at the unpacked archive. System paths are searched as
# a fallback.
#
# Variables set:
#   Cutensor_INCLUDE_DIR, Cutensor_LIBRARIES, Cutensor_FOUND
#
# Cache option:
#   Cutensor_USE_STATIC (default OFF) -- prefer libcutensor_static.a over
#   libcutensor.so. If only one variant is present on disk it is picked
#   automatically regardless of this setting.
#
# Target:
#   cutensor (INTERFACE) -- link this. When the resolved library is the
#   static archive, CUDA::culibos, dl, Threads, and rt (Linux) are linked in.

if(Cutensor_INCLUDE_DIR AND Cutensor_LIBRARIES)
  set(Cutensor_FIND_QUIETLY TRUE)
endif()

# Carry over legacy CUTENSOR_ROOT spelling.
if(DEFINED CUTENSOR_ROOT AND NOT DEFINED Cutensor_ROOT)
  set(Cutensor_ROOT "${CUTENSOR_ROOT}")
endif()

option(Cutensor_USE_STATIC "Prefer libcutensor_static.a over libcutensor.so" OFF)
if(Cutensor_USE_STATIC)
  set(_cutensor_names cutensor_static cutensor)
else()
  set(_cutensor_names cutensor cutensor_static)
endif()

find_path(Cutensor_INCLUDE_DIR
  NAMES cutensor.h
  HINTS
    ${Cutensor_ROOT}/include
    $ENV{Cutensor_ROOT}/include
    $ENV{CUTENSOR_ROOT}/include
    ENV CPATH
    ENV C_INCLUDE_PATH
    ENV CPLUS_INCLUDE_PATH
    /usr/include
    /usr/local/include
    /opt/local/include
  DOC "Include directory for cuTENSOR"
)

# cuTENSOR archive layouts:
#   <root>/lib/<cuda-major>/{libcutensor.so,libcutensor_static.a}  (legacy)
#   <root>/lib/{libcutensor.so,libcutensor_static.a}               (>=2.0)
set(_cutensor_libdir_hints
  ${Cutensor_ROOT}/lib
  $ENV{Cutensor_ROOT}/lib
  $ENV{CUTENSOR_ROOT}/lib
  ${Cutensor_INCLUDE_DIR}/../lib
)
if(DEFINED CUDAToolkit_VERSION_MAJOR)
  list(PREPEND _cutensor_libdir_hints
    ${Cutensor_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}
    $ENV{Cutensor_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}
    $ENV{CUTENSOR_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}
    ${Cutensor_INCLUDE_DIR}/../lib/${CUDAToolkit_VERSION_MAJOR}
  )
endif()
list(APPEND _cutensor_libdir_hints
  ENV LIBRARY_PATH
  ENV LD_LIBRARY_PATH
  /usr/lib
  /usr/local/lib
  /opt/local/lib
)

find_library(Cutensor_LIBRARIES
  NAMES ${_cutensor_names}
  HINTS ${_cutensor_libdir_hints}
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
      CUDA::culibos
      Threads::Threads
      ${CMAKE_DL_LIBS}
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_link_libraries(cutensor INTERFACE rt)
    endif()
  endif()
endif()
