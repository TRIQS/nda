// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Includes all MPI relevant headers.
 */

#pragma once

#ifndef NDA_HAVE_MPI
#error "MPI support is not enabled in this build of nda. Please configure and install nda with -DMPISupport=ON"
#endif

#include "./mpi/broadcast.hpp"
#include "./mpi/gather.hpp"
#include "./mpi/reduce.hpp"
#include "./mpi/scatter.hpp"
#include "./mpi/utils.hpp"
