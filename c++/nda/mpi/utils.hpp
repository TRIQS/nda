// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides various utility functions used by the MPI interface of nda.
 */

#pragma once

#include "../concepts.hpp"
#include "../exceptions.hpp"

#include <mpi/mpi.hpp>

#include <string>

namespace nda::detail {

  // Check if the layout of an array/view is contiguous with positive strides, otherwise throw an exception.
  template <nda::Array A>
  void check_layout_mpi_compatible(const A &a, const std::string &func) {
    if (not a.is_contiguous() or not a.has_positive_strides())
      NDA_RUNTIME_ERROR << "Error in function " << func << ": Array is not contiguous with positive strides";
  }

  // Check if the ranks of arrays/views are the same on all processes.
  template <nda::Array A>
  bool have_mpi_equal_ranks(const A &a, const mpi::communicator &comm) {
    return mpi::all_equal(a.rank, comm);
  }

  // Check if the shape of arrays/views are the same on all processes.
  template <nda::Array A>
  bool have_mpi_equal_shapes(const A &a, const mpi::communicator &comm) {
    return have_mpi_equal_ranks(a, comm) && mpi::all_equal(a.shape(), comm);
  }

} // namespace nda::detail
