// Copyright (c) 2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides an MPI broadcast function for nda::basic_array or nda::basic_array_view types.
 */

#pragma once

#include "./utils.hpp"
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
   * array/view.
   *
   * Throws an exception, if
   * - a given view does not have the correct shape,
   * - the array/view is not contiguous with positive strides or
   * - one of the MPI calls fails.
   *
   * @code{.cpp}
   * // create an array on all processes
   * nda::array<int, 2> A(3, 4);
   *
   * // ...
   * // fill array on root process
   * // ...
   *
   * // broadcast the array to all processes
   * mpi::broadcast(A);
   * @endcode
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
    detail::check_layout_mpi_compatible(a, "mpi_broadcast");
    auto dims = a.shape();
    mpi::broadcast(dims, comm, root);
    if (comm.rank() != root) { resize_or_check_if_view(a, dims); }
    mpi::broadcast_range(std::span{a.data(), static_cast<std::size_t>(a.size())}, comm, root);
  }

} // namespace nda
