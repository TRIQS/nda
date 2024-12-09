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
 * @brief Provides an MPI gather function for nda::basic_array or nda::basic_array_view types.
 */

#pragma once

#include "./utils.hpp"
#include "../basic_functions.hpp"
#include "../concepts.hpp"
#include "../declarations.hpp"
#include "../layout/range.hpp"
#include "../macros.hpp"
#include "../stdutil/array.hpp"
#include "../traits.hpp"

#include <mpi/mpi.hpp>

#include <cstddef>
#include <functional>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>

namespace nda::detail {

  // Helper function to get the shape and total size of the gathered array/view.
  template <typename A>
    requires(is_regular_or_view_v<A> and std::decay_t<A>::is_stride_order_C())
  auto mpi_gather_shape_impl(A const &a, mpi::communicator comm, int root, bool all) {
    auto dims          = a.shape();
    dims[0]            = mpi::all_reduce(dims[0], comm);
    auto gathered_size = std::accumulate(dims.begin(), dims.end(), 1l, std::multiplies<>());
    if (!all && comm.rank() != root) dims = nda::stdutil::make_initialized_array<dims.size()>(0l);
    return std::make_pair(dims, gathered_size);
  }

} // namespace nda::detail

namespace nda {

  /**
   * @addtogroup av_mpi
   * @{
   */

  /**
   * @brief Implementation of an MPI gather for nda::basic_array or nda::basic_array_view types using a C-style API.
   *
   * @details The function gathers C-ordered input arrays/views from all processes in the given communicator and
   * makes the result available on the root process (`all == false`) or on all processes (`all == true`). The
   * arrays/views are joined along the first dimension.
   *
   * It is expected that all input arrays/views have the same shape on all processes except for the first dimension. The
   * function throws an exception, if
   * - the input array/view is not contiguous with positive strides,
   * - the output array/view is not contiguous with positive strides on receiving ranks,
   * - the output view does not have the correct shape on receiving ranks or
   * - any of the MPI calls fails.
   *
   * The input arrays/views are simply concatenated along their first dimension. The content of the output array/view
   * depends on the MPI rank and whether it receives the data or not:
   * - On receiving ranks, it contains the gathered data and has a shape that is the same as the shape of the input
   * array/view except for the first dimension, which is the sum of the extents of all input arrays/views along the
   * first dimension.
   * - On non-receiving ranks, the output array/view is ignored and left unchanged.
   *
   * If `mpi::has_env` is false or if the communicator size is < 2, it simply copies the input array/view to the output
   * array/view.
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
  void mpi_gather_capi(A1 const &a_in, A2 &&a_out, mpi::communicator comm = {}, int root = 0, bool all = false) { // NOLINT
    // check the shape of the input arrays/views
    EXPECTS_WITH_MESSAGE(detail::have_mpi_equal_shapes(a_in(nda::range(1), nda::ellipsis{}), comm),
                         "Error in nda::mpi_gather_capi: Shapes of arrays/views must be equal save the first one");

    // simply copy if there is no active MPI environment or if the communicator size is < 2
    if (not mpi::has_env || comm.size() < 2) {
      a_out = a_in;
      return;
    }

    // check if the input arrays/views can be used in the MPI call
    detail::check_layout_mpi_compatible(a_in, "mpi_gather_capi");

    // get output shape, resize or check the output array/view and prepare output span
    auto [dims, gathered_size] = detail::mpi_gather_shape_impl(a_in, comm, root, all);
    auto a_out_span            = std::span{a_out.data(), 0};
    if (all || (comm.rank() == root)) {
      // check if the output array/view can be used in the MPI call
      detail::check_layout_mpi_compatible(a_out, "mpi_gather_capi");

      // resize/check the size of the output array/view
      resize_or_check_if_view(a_out, dims);

      // prepare the output span
      a_out_span = std::span{a_out.data(), static_cast<std::size_t>(a_out.size())};
    }

    // gather the data
    auto a_in_span = std::span{a_in.data(), static_cast<std::size_t>(a_in.size())};
    mpi::gather_range(a_in_span, a_out_span, gathered_size, comm, root, all);
  }

  /**
   * @brief Implementation of a lazy MPI gather for nda::basic_array or nda::basic_array_view types.
   *
   * @details This function is lazy, i.e. it returns an mpi::lazy<mpi::tag::gather, A> object without performing the
   * actual MPI operation. Since the returned object models an nda::ArrayInitializer, it can be used to
   * initialize/assign to nda::basic_array and nda::basic_array_view objects.
   *
   * The behavior is otherwise similar to nda::mpi_gather.
   *
   * @warning MPI calls are done in the `invoke` and `shape` methods of the `mpi::lazy` object. If one rank calls one of
   * these methods, all ranks in the communicator need to call the same method. Otherwise, the program will deadlock.
   *
   * @tparam A nda::basic_array or nda::basic_array_view type with C-layout.
   * @param a Array/view to be gathered.
   * @param comm `mpi::communicator` object.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather.
   * @return An mpi::lazy<mpi::tag::gather, A> object modelling an nda::ArrayInitializer.
   */
  template <typename A>
    requires(is_regular_or_view_v<A> and std::decay_t<A>::is_stride_order_C())
  auto lazy_mpi_gather(A &&a, mpi::communicator comm = {}, int root = 0, bool all = false) {
    return mpi::lazy<mpi::tag::gather, A>{std::forward<A>(a), comm, root, all};
  }

  /**
   * @brief Implementation of an MPI gather for nda::basic_array or nda::basic_array_view types.
   *
   * @details The function gathers C-ordered input arrays/views from all processes in the given communicator and
   * makes the result available on the root process (`all == false`) or on all processes (`all == true`). The
   * arrays/views are joined along the first dimension.
   *
   * It simply constructs an empty array and then calls nda::mpi_gather_capi.
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
    mpi_gather_capi(a, a_out, comm, root, all);
    return a_out;
  }

  /** @} */

} // namespace nda

/**
 * @ingroup av_mpi
 * @brief Specialization of the `mpi::lazy` class for nda::Array types and the `mpi::tag::gather` tag.
 *
 * @details An object of this class is returned when gathering nda::Array objects across multiple MPI processes.
 *
 * It models an nda::ArrayInitializer, that means it can be used to initialize and assign to nda::basic_array and
 * nda::basic_array_view objects. The result will be a concatenation of the input arrays/views along their first
 * dimension.
 *
 * See nda::lazy_mpi_gather for an example and more information.
 *
 * @tparam A nda::Array type to be gathered.
 */
template <nda::Array A>
struct mpi::lazy<mpi::tag::gather, A> {
  /// Value type of the array/view.
  using value_type = typename std::decay_t<A>::value_type;

  /// Type of the array/view stored in the lazy object.
  using stored_type = A;

  /// Array/View to be gathered.
  stored_type rhs;

  /// MPI communicator.
  mpi::communicator comm;

  /// MPI root process.
  const int root{0}; // NOLINT (const is fine here)

  /// Should all processes receive the result.
  const bool all{false}; // NOLINT (const is fine here)

  /**
   * @brief Compute the shape of the nda::ArrayInitializer object.
   *
   * @details The input arrays/views are simply concatenated along their first dimension. The shape of the initializer
   * object depends on the MPI rank and whether it receives the data or not:
   * - On receiving ranks, the shape is the same as the shape of the input array/view except for the first dimension,
   * which is the sum of the extents of all input arrays/views along the first dimension.
   * - On non-receiving ranks, the shape is empty, i.e. `(0,0,...,0)`.
   *
   * @warning This makes an MPI call.
   *
   * @return Shape of the nda::ArrayInitializer object.
   */
  [[nodiscard]] auto shape() const { return nda::detail::mpi_gather_shape_impl(rhs, comm, root, all).first; }

  /**
   * @brief Execute the lazy MPI operation and write the result to a target array/view.
   *
   * @details The data will be gathered directly into the memory handle of the target array/view.
   *
   * It is expected that all input arrays/views have the same shape on all processes except for the first dimension. The
   * function throws an exception, if
   * - the input array/view is not contiguous with positive strides,
   * - the target array/view is not contiguous with positive strides on receiving ranks,
   * - a target view does not have the correct shape on receiving ranks or
   * - if any of the MPI calls fails.
   *
   * @tparam T nda::Array type with C-layout.
   * @param target Target array/view.
   */
  template <nda::Array T>
    requires(std::decay_t<T>::is_stride_order_C())
  void invoke(T &&target) const { // NOLINT (temporary views are allowed here)
    nda::mpi_gather_capi(rhs, target, comm, root, all);
  }
};
