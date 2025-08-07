// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides an MPI scatter function for nda::basic_array or nda::basic_array_view types.
 */

#pragma once

#include "./utils.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../traits.hpp"

#include <mpi/mpi.hpp>

#include <cstddef>
#include <functional>
#include <numeric>
#include <span>
#include <tuple>
#include <type_traits>

namespace nda::detail {

  // Helper function to get the shape and total size of the scattered array/view as well as the stride along the first
  // dimension.
  template <typename A>
    requires(is_regular_or_view_v<A> and std::decay_t<A>::is_stride_order_C())
  auto mpi_scatter_shape_impl(A const &a, mpi::communicator comm, int root) {
    auto dims = a.shape();
    mpi::broadcast(dims, comm, root);
    auto scattered_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
    auto stride0        = scattered_size / dims[0];
    dims[0]             = mpi::chunk_length(dims[0], comm.size(), comm.rank());
    return std::make_tuple(dims, scattered_size, stride0);
  }

} // namespace nda::detail

namespace nda {

  /**
   * @addtogroup av_mpi
   * @{
   */

  /**
   * @brief Implementation of an MPI scatter for nda::basic_array or nda::basic_array_view types that scatters directly
   * into an existing array/view.
   *
   * @details The function scatters a C-ordered input array/view from a root process across all processes in the given
   * communicator. The array/view is chunked into equal parts along the first dimension using `mpi::chunk_length`.
   *
   * It is expected that all input arrays/views have the same rank on all processes. The function throws an exception,
   * if
   * - the input array/view is not contiguous with positive strides on the root process,
   * - an output array/view is not contiguous with positive strides or
   * - an output view does not have the correct shape.
   *
   * The actual scattering is done by calling `mpi::scatter_range`. The input array/view on the root process is chunked
   * along the first dimension into equal (as much as possible) parts using `mpi::chunk_length`. If the extent of the
   * input array along the first dimension is not divisible by the number of processes, processes with lower ranks will
   * receive more data than processes with higher ranks.
   *
   * @note Scattering is only supported for contiguous arrays/views with positive strides and with MPI compatible value
   * types.
   *
   * @tparam A1 nda::basic_array or nda::basic_array_view type with C-layout.
   * @tparam A2 nda::basic_array or nda::basic_array_view type with C-layout.
   * @param a_in Array/view to be scattered.
   * @param a_out Array/view to scatter into.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   */
  template <typename A1, typename A2>
    requires(is_regular_or_view_v<A1> and std::decay_t<A1>::is_stride_order_C()
             and is_regular_or_view_v<A2> and std::decay_t<A2>::is_stride_order_C())
  void mpi_scatter_into(A1 const &a_in, A2 &&a_out, mpi::communicator comm = {}, int root = 0) { // NOLINT
    // check the ranks of the input arrays/views
    EXPECTS_WITH_MESSAGE(detail::have_mpi_equal_ranks(a_in, comm), "Error in nda::mpi_scatter_into: Ranks of arrays/views must be equal")

    // check if the input and output arrays/views can be used in the MPI call
    if (comm.rank() == root) detail::check_layout_mpi_compatible(a_in, "mpi_scatter_into");
    detail::check_layout_mpi_compatible(a_out, "mpi_scatter_into");

    // get output shape and resize or check the output array/view
    auto [dims, scattered_size, stride0] = detail::mpi_scatter_shape_impl(a_in, comm, root);
    resize_or_check_if_view(a_out, dims);

    // scatter the data
    auto a_out_span = std::span{a_out.data(), static_cast<std::size_t>(a_out.size())};
    auto a_in_span  = std::span{a_in.data(), static_cast<std::size_t>(a_in.size())};
    mpi::scatter_range(a_in_span, a_out_span, scattered_size, comm, root, stride0);
  }

  /**
   * @brief Implementation of an MPI scatter for nda::basic_array or nda::basic_array_view types.
   *
   * @details The function scatters a C-ordered input array/view from a root process across all processes in the given
   * communicator. The array/view is chunked into equal parts along the first dimension using `mpi::chunk_length`.
   *
   * It simply constructs an empty array and then calls nda::mpi_scatter_into.
   *
   * See @ref ex6_p3 for an example.
   *
   * @tparam A nda::basic_array or nda::basic_array_view type.
   * @param a Array/view to be scattered.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @return An nda::basic_array object with the result of the scattering.
   */
  template <typename A>
    requires(is_regular_or_view_v<A> and std::decay_t<A>::is_stride_order_C())
  auto mpi_scatter(A const &a, mpi::communicator comm = {}, int root = 0) {
    using return_t = get_regular_t<A>;
    return_t a_out;
    mpi::scatter_into(a, a_out, comm, root);
    return a_out;
  }

  /** @} */

} // namespace nda
