#  Copyright Olivier Parcollet 2010.
#  Copyright Simons Foundation 2019
#    Author: Nils Wentzell

#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

#
# This module looks for magma.
# It sets up : Magma_INCLUDE_DIR, Magma_LIBRARIES
# Use Magma3_ROOT to specify a particular location
#

if(Magma_INCLUDE_DIR AND Magma_LIBRARIES)
  set(Magma_FIND_QUIETLY TRUE)
endif()

find_path(Magma_INCLUDE_DIR
  NAMES magma_v2.h
  HINTS
    ${Magma_ROOT}/include
    $ENV{Magma_ROOT}/include
    $ENV{Magma_BASE}/include
    ENV CPATH
    ENV C_INCLUDE_PATH
    ENV CPLUS_INCLUDE_PATH
    ENV OBJC_INCLUDE_PATH
    ENV OBJCPLUS_INCLUDE_PATH
    /usr/include
    /usr/local/include
    /opt/local/include
    /sw/include
  DOC "Include Directory for Magma"
)

find_library(Magma_LIBRARIES
  NAMES magma
  HINTS
    ${Magma_INCLUDE_DIR}/../lib
    ${Magma_ROOT}/lib
    $ENV{Magma_ROOT}/lib
    $ENV{Magma_BASE}/lib
    ENV LIBRARY_PATH
    ENV LD_LIBRARY_PATH
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
  DOC "Magma library"
)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Magma DEFAULT_MSG Magma_LIBRARIES Magma_INCLUDE_DIR)

mark_as_advanced(Magma_INCLUDE_DIR Magma_LIBRARIES)

# Interface target
# We refrain from creating an imported target since those cannot be exported
add_library(magma INTERFACE)
target_link_libraries(magma INTERFACE ${Magma_LIBRARIES})
target_include_directories(magma SYSTEM INTERFACE ${Magma_INCLUDE_DIR})
