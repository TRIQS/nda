// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides an MPI broadcast function for nda::basic_array or nda::basic_array_view types.
 */

#pragma once

#include "../basic_functions.hpp"
#include "../traits.hpp"

#include <mpi/mpi.hpp>

#include <cstddef>
#include <span>

namespace nda {

  /**
   * @ingroup av_mpi
   * @brief Implementation of an MPI broadcast for nda::basic_array or nda::basic_array_view types.
   *
   * @details For the root process, the array/view is broadcasted to all other processes. For non-root processes, the
   * array/view is resized/checked to match the broadcasted dimensions and the data is written into the given
   * array/view. The actual broadcasting is done by calling `mpi::broadcast_range`.
   *
   * It throws an exception, if a given view does not have the correct shape.
   *
   * See @ref ex6_p1 for an example.
   *
   * @note If the array/view is contiguous with positive strides and if the value type is MPI compatible, the data is
   * broadcasted using a single `MPI_Bcast` call. Otherwise, the data is broadcasted element-wise which can have
   * considerable performance implications. Consider copying the data into a contiguous array/view before broadcasting.
   *
   * @tparam A nda::basic_array or nda::basic_array_view type.
   * @param a Array/view to be broadcasted from/into.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   */
  template <typename A>
  void mpi_broadcast(A &&a, mpi::communicator comm = {}, int root = 0) // NOLINT (temporary views are allowed here)
    requires(is_regular_or_view_v<A>)
  {
    // broadcast the shape of the input array/view on the root process
    auto dims = a.shape();
    mpi::broadcast(dims, comm, root);

    // resize or check the output array/view on non-root processes
    if (comm.rank() != root) resize_or_check_if_view(a, dims);

    // call mpi::broadcast_range with a span if the array/view is contiguous with positive strides
    if (a.is_contiguous() and a.has_positive_strides()) {
      auto a_span = std::span{a.data(), static_cast<std::size_t>(a.size())};
      mpi::broadcast_range(a_span, comm, root);
    } else {
      mpi::broadcast_range(a, comm, root);
    }
  }

} // namespace nda
