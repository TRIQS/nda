// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides an MPI reduce function for nda::basic_array or nda::basic_array_view types.
 */

#pragma once

#include "./utils.hpp"
#include "../basic_functions.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../map.hpp"
#include "../traits.hpp"

#include <mpi.h>
#include <mpi/mpi.hpp>

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

namespace nda {

  /**
   * @addtogroup av_mpi
   * @{
   */

  /**
   * @brief Implementation of an MPI reduce for nda::basic_array or nda::basic_array_view types that reduces directly
   * into an existing array/view.
   *
   * @details The function reduces input arrays/views from all processes in the given communicator and makes the result
   * available on the root process (`all == false`) or on all processes (`all == true`).
   *
   * It is expected that all input arrays/views have the same shape on all processes. The function throws an exception,
   * if an output view does not have the correct shape on receiving ranks.
   *
   * The actual reduction is done by calling `mpi::reduce_range`. The content of the output array/view depends on the
   * MPI rank and whether it receives the data or not:
   * - On receiving ranks, it contains the reduced data and has a shape that is the same as the shape of the input
   * array/view.
   * - On non-receiving ranks, the output array/view is ignored and left unchanged.
   *
   * @note If the input/output arrays/views are contiguous with positive strides and if the value type is MPI
   * compatible, the data is reduced using a single `MPI_Reduce` or `MPI_Allreduce` call. Otherwise, the data is reduced
   * element-wise, which can have considerable performance implications. Consider copying the data into a contiguous
   * array/view before reducing.
   *
   * @tparam A1 nda::basic_array or nda::basic_array_view type.
   * @tparam A2 nda::basic_array or nda::basic_array_view type.
   * @param a_in Array/view to be reduced.
   * @param a_out Array/view to reduce into.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op MPI reduction operation.
   */
  template <typename A1, typename A2>
    requires(is_regular_or_view_v<A1> && is_regular_or_view_v<A2>)
  void mpi_reduce_into(A1 const &a_in, A2 &&a_out, mpi::communicator comm = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) { // NOLINT
    // check the shape of the input arrays/views
    EXPECTS_WITH_MESSAGE(detail::have_mpi_equal_shapes(a_in, comm), "Error in nda::mpi_reduce_into: Shapes of arrays/views must be equal");

    // resize or check the output array/view on receiving ranks
    bool const receives = (all || (comm.rank() == root));
    if (receives) resize_or_check_if_view(a_out, a_in.shape());

    // call mpi::reduce_range with a span if the input and output arrays/views are contiguous with positive strides
    // (on non-receiving ranks, the output should always be contiguous since it is not used)
    if (a_in.is_contiguous() and a_in.has_positive_strides()) {
      auto a_in_span = std::span{a_in.data(), static_cast<std::size_t>(a_in.size())};
      if ((a_out.is_contiguous() and a_out.has_positive_strides()) || !receives) {
        auto a_out_span = std::span{a_out.data(), static_cast<std::size_t>(a_out.size())};
        mpi::reduce_range(a_in_span, a_out_span, comm, root, all, op);
      } else {
        mpi::reduce_range(a_in_span, a_out, comm, root, all, op);
      }
    } else {
      if ((a_out.is_contiguous() and a_out.has_positive_strides()) || !receives) {
        auto a_out_span = std::span{a_out.data(), static_cast<std::size_t>(a_out.size())};
        mpi::reduce_range(a_in, a_out_span, comm, root, all, op);
      } else {
        mpi::reduce_range(a_in, a_out, comm, root, all, op);
      }
    }
  }

  /**
   * @brief Implementation of an MPI reduce for nda::basic_array or nda::basic_array_view types.
   *
   * @details The function reduces input arrays/views from all processes in the given communicator and makes the result
   * available on the root process (`all == false`) or on all processes (`all == true`).
   *
   * It first default constructs an nda::basic_array object on the heap with its value type equal to the return type of
   * `reduce(std::declval<get_value_t<A>>())` and the same rank and algebra as the input array/view. On receiving ranks,
   * the output array is then resized to the shape of the input array/view.
   *
   * The actual reduction is done by calling nda::mpi_reduce_into with the input array/view and the constructed output
   * array. The content of the returned array depends on the MPI rank and whether it receives the data or not:
   * - On receiving ranks, it contains the reduced data.
   * - On non-receiving ranks, the array is empty.
   *
   * See @ref ex6_p4 for an example.
   *
   * @note If the input arrays/views are contiguous with positive strides and if the value type is MPI compatible, the
   * data is reduced using a single `MPI_Reduce` or `MPI_Allreduce` call. Otherwise, the data is reduced element-wise,
   * which can have considerable performance implications. Consider copying the data into a contiguous array/view before
   * reducing.
   *
   * @tparam A nda::basic_array or nda::basic_array_view type.
   * @param a Array/view to be reduced.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op MPI reduction operation.
   * @return An nda::basic_array object with the result of the reduction.
   */
  template <typename A>
    requires(is_regular_or_view_v<A>)
  auto mpi_reduce(A const &a, mpi::communicator comm = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    using value_t  = std::remove_cvref_t<decltype(mpi::reduce(std::declval<get_value_t<A>>()))>;
    using return_t = basic_array<value_t, get_rank<A>, typename A::layout_policy_t::contiguous_t, get_algebra<A>, heap<>>;
    return_t a_out;
    mpi_reduce_into(a, a_out, comm, root, all, op);
    return a_out;
  }

  /** @} */

} // namespace nda
