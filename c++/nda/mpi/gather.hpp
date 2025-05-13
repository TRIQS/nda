// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides an MPI gather function for nda::basic_array or nda::basic_array_view types.
 */

#pragma once

#include "./utils.hpp"
#include "../basic_functions.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../stdutil/array.hpp"
#include "../traits.hpp"

#include <mpi/mpi.hpp>

#include <cstddef>
#include <span>
#include <type_traits>

namespace nda::detail {

  // Helper function to get the shape of the gathered array/view.
  template <typename A>
    requires(is_regular_or_view_v<A> and std::decay_t<A>::is_stride_order_C())
  auto mpi_gather_shape_impl(A const &a, mpi::communicator comm, int root, bool all) {
    auto dims = a.shape();
    dims[0]   = mpi::all_reduce(dims[0], comm);
    if (!all && comm.rank() != root) dims = nda::stdutil::make_initialized_array<dims.size()>(0l);
    return dims;
  }

} // namespace nda::detail

namespace nda {

  /**
   * @addtogroup av_mpi
   * @{
   */

  /**
   * @brief Implementation of an MPI gather for nda::basic_array or nda::basic_array_view types that gathers directly
   * into an existing array/view.
   *
   * @details The function gathers C-ordered input arrays/views from all processes in the given communicator and
   * makes the result available on the root process (`all == false`) or on all processes (`all == true`). The
   * arrays/views are joined along the first dimension.
   *
   * It is expected that all input arrays/views have the same shape on all processes except for the first dimension. The
   * function throws an exception if
   * - an input array/view is not contiguous with positive strides,
   * - an output array/view is not contiguous with positive strides on receiving ranks or
   * - if an output view does not have the correct shape on receiving ranks.
   *
   * The actual gathering is done by calling `mpi::gather_range`. The input arrays/views are simply concatenated along
   * their first dimension. The content of the output array/view depends on the MPI rank and whether it receives the
   * data or not:
   * - On receiving ranks, it contains the gathered data and has a shape that is the same as the shape of the input
   * array/view except for the first dimension, which is the sum of the extents of all input arrays/views along the
   * first dimension.
   * - On non-receiving ranks, the output array/view is ignored and left unchanged.
   *
   * @note Gathering is only supported for contiguous arrays/views with positive strides and with MPI compatible value
   * types.
   *
   * @tparam A1 nda::basic_array or nda::basic_array_view type with C-layout.
   * @tparam A2 nda::basic_array or nda::basic_array_view type with C-layout.
   * @param a_in Array/view to be gathered.
   * @param a_out Array/view to gather into.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather.
   */
  template <typename A1, typename A2>
    requires(is_regular_or_view_v<A1> and std::decay_t<A1>::is_stride_order_C()
             and is_regular_or_view_v<A2> and std::decay_t<A2>::is_stride_order_C())
  void mpi_gather_into(A1 const &a_in, A2 &&a_out, mpi::communicator comm = {}, int root = 0, bool all = false) { // NOLINT
    // check the shape of the input arrays/views
    EXPECTS_WITH_MESSAGE(detail::have_mpi_equal_shapes(a_in(nda::range(1), nda::ellipsis{}), comm),
                         "Error in nda::mpi_gather_into: Shapes of arrays/views must be equal save the first one");

    // get output shape
    auto dims = detail::mpi_gather_shape_impl(a_in, comm, root, all);

    // check if the input and output arrays/views can be used in the MPI call and resize or check the output array/view
    detail::check_layout_mpi_compatible(a_in, "mpi_gather_into");
    bool const receives = (all || (comm.rank() == root));
    if (receives) {
      detail::check_layout_mpi_compatible(a_out, "mpi_gather_into");
      resize_or_check_if_view(a_out, dims);
    }

    // gather the data
    auto a_in_span  = std::span{a_in.data(), static_cast<std::size_t>(a_in.size())};
    auto a_out_span = std::span{a_out.data(), static_cast<std::size_t>(a_out.size())};
    mpi::gather_range(a_in_span, a_out_span, comm, root, all);
  }

  /**
   * @brief Implementation of an MPI gather for nda::basic_array or nda::basic_array_view types.
   *
   * @details The function gathers C-ordered input arrays/views from all processes in the given communicator and
   * makes the result available on the root process (`all == false`) or on all processes (`all == true`). The
   * arrays/views are joined along the first dimension.
   *
   * It simply constructs an empty array and then calls nda::mpi_gather_into.
   *
   * See @ref ex6_p2 for examples.
   *
   * @tparam A nda::basic_array or nda::basic_array_view type with C-layout.
   * @param a Array/view to be gathered.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather.
   * @return An nda::basic_array object with the result of the gathering.
   */
  template <typename A>
    requires(is_regular_or_view_v<A> and std::decay_t<A>::is_stride_order_C())
  auto mpi_gather(A const &a, mpi::communicator comm = {}, int root = 0, bool all = false) {
    using return_t = get_regular_t<A>;
    return_t a_out;
    mpi::gather_into(a, a_out, comm, root, all);
    return a_out;
  }

  /** @} */

} // namespace nda
