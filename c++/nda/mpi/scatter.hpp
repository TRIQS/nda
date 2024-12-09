// Copyright (c) 2020-2024 Simons Foundation
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
 * @brief Provides an MPI scatter function for nda::basic_array or nda::basic_array_view types.
 */

#pragma once

#include "./utils.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../macros.hpp"
#include "../traits.hpp"

#include <mpi.h>
#include <mpi/mpi.hpp>

#include <cstddef>
#include <functional>
#include <numeric>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nda::detail {

  // Helper function to get the shape and total size of the scattered array/view.
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
   * @brief Implementation of an MPI scatter for nda::basic_array or nda::basic_array_view types using a C-style API.
   *
   * @details The function scatters a C-ordered input array/view from a root process across all processes in the given
   * communicator. The array/view is chunked into equal parts along the first dimension using `mpi::chunk_length`.
   *
   * It is expected that all input arrays/views have the same rank on all processes. The function throws an exception,
   * if
   * - the input array/view is not contiguous with positive strides on the root process,
   * - the output array/view is not contiguous with positive strides,
   * - the output view does not have the correct shape or
   * - any of the MPI calls fails.
   *
   * The input array/view on the root process is chunked along the first dimension into equal (as much as possible)
   * parts using `mpi::chunk_length`. If the extent of the input array along the first dimension is not divisible by the
   * number of processes, processes with lower ranks will receive more data than processes with higher ranks.
   *
   * If `mpi::has_env` is false or if the communicator size is < 2, it simply copies the input array/view to the output
   * array/view.
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
  void mpi_scatter_capi(A1 const &a_in, A2 &&a_out, mpi::communicator comm = {}, int root = 0) { // NOLINT
    // check the ranks of the input arrays/views
    EXPECTS_WITH_MESSAGE(detail::have_mpi_equal_ranks(a_in, comm), "Error in nda::mpi_scatter_capi: Ranks of arrays/views must be equal")

    // simply copy if there is no active MPI environment or if the communicator size is < 2
    if (not mpi::has_env || comm.size() < 2) {
      a_out = a_in;
      return;
    }

    // check if the input and output arrays/views can be used in the MPI call
    if (comm.rank() == root) detail::check_layout_mpi_compatible(a_in, "mpi_scatter_capi");
    detail::check_layout_mpi_compatible(a_out, "mpi_scatter_capi");

    // get output shape and resize or check the output array/view
    auto [dims, scattered_size, stride0] = detail::mpi_scatter_shape_impl(a_in, comm, root);
    resize_or_check_if_view(a_out, dims);

    // scatter the data
    auto a_out_span = std::span{a_out.data(), static_cast<std::size_t>(a_out.size())};
    auto a_in_span  = std::span{a_in.data(), static_cast<std::size_t>(a_in.size())};
    mpi::scatter_range(a_in_span, a_out_span, scattered_size, comm, root, stride0);
  }

  /**
   * @brief Implementation of a lazy MPI scatter for nda::basic_array or nda::basic_array_view types.
   *
   * @details This function is lazy, i.e. it returns an mpi::lazy<mpi::tag::scatter, A> object without performing the
   * actual MPI operation. Since the returned object models an nda::ArrayInitializer, it can be used to
   * initialize/assign to nda::basic_array and nda::basic_array_view objects.
   *
   * The behavior is otherwise similar to nda::mpi_scatter.
   *
   * @warning MPI calls are done in the `invoke` and `shape` methods of the `mpi::lazy` object. If one rank calls one of
   * these methods, all ranks in the communicator need to call the same method. Otherwise, the program will deadlock.
   *
   * @tparam A nda::basic_array or nda::basic_array_view type.
   * @param a Array/view to be scattered.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @return An mpi::lazy<mpi::tag::scatter, A> object modelling an nda::ArrayInitializer.
   */
  template <typename A>
    requires(is_regular_or_view_v<A> and std::decay_t<A>::is_stride_order_C())
  auto lazy_mpi_scatter(A &&a, mpi::communicator comm = {}, int root = 0) {
    return mpi::lazy<mpi::tag::scatter, A>{std::forward<A>(a), comm, root, true};
  }

  /**
   * @brief Implementation of an MPI scatter for nda::basic_array or nda::basic_array_view types.
   *
   * @details The function scatters a C-ordered input array/view from a root process across all processes in the given
   * communicator. The array/view is chunked into equal parts along the first dimension using `mpi::chunk_length`.
   *
   * It simply constructs an empty array and then calls nda::mpi_scatter_capi.
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
    mpi_scatter_capi(a, a_out, comm, root);
    return a_out;
  }

  /** @} */

} // namespace nda

/**
 * @ingroup av_mpi
 * @brief Specialization of the `mpi::lazy` class for nda::Array types and the `mpi::tag::scatter` tag.
 *
 * @details An object of this class is returned when scattering nda::Array objects across multiple MPI processes.
 *
 * It models an nda::ArrayInitializer, that means it can be used to initialize and assign to nda::basic_array and
 * nda::basic_array_view objects. The input array/view on the root process will be chunked along the first dimension
 * into equal parts using `mpi::chunk_length` and scattered across all processes in the communicator.
 *
 * See nda::mpi_scatter for an example and more information.
 *
 * @tparam A nda::Array type to be scattered.
 */
template <nda::Array A>
struct mpi::lazy<mpi::tag::scatter, A> {
  /// Value type of the array/view.
  using value_type = typename std::decay_t<A>::value_type;

  /// Type of the array/view stored in the lazy object.
  using stored_type = A;

  /// Array/View to be scattered.
  stored_type rhs;

  /// MPI communicator.
  mpi::communicator comm;

  /// MPI root process.
  const int root{0}; // NOLINT (const is fine here)

  /// Should all processes receive the result. (doesn't make sense for scatter)
  const bool all{false}; // NOLINT (const is fine here)

  /**
   * @brief Compute the shape of the nda::ArrayInitializer object.
   *
   * @details The input array/view on the root process is chunked along the first dimension into equal (as much as
   * possible) parts using `mpi::chunk_length`.
   *
   * If the extent of the input array along the first dimension is not divisible by the number of processes, processes
   * with lower ranks will receive more data than processes with higher ranks.
   *
   * @warning This makes an MPI call.
   *
   * @return Shape of the nda::ArrayInitializer object.
   */
  [[nodiscard]] auto shape() const { return std::get<0>(nda::detail::mpi_scatter_shape_impl(rhs, comm, root)); }

  /**
   * @brief Execute the lazy MPI operation and write the result to a target array/view.
   *
   * @details The data will be scattered directly into the memory handle of the target array/view.
   *
   * It is expected that all input arrays/views have the same rank on all processes. The function throws an exception,
   * if
   * - the input array/view on the root process is not contiguous with positive strides,
   * - the target array/view is not contiguous with positive,
   * - a target view does not have the correct shape or
   * - if any of the MPI calls fails.
   *
   * @tparam T nda::Array type with C-layout.
   * @param target Target array/view.
   */
  template <nda::Array T>
    requires(std::decay_t<T>::is_stride_order_C())
  void invoke(T &&target) const { // NOLINT (temporary views are allowed here)
    nda::mpi_scatter_capi(rhs, target, comm, root);
  }
};
