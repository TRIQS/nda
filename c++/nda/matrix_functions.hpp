// Copyright (c) 2019--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides functions to create and manipulate matrices, i.e. arrays/view with 'M' algebra.
 */

#pragma once

#include "./accessors.hpp"
#include "./concepts.hpp"
#include "./declarations.hpp"
#include "./layout/policies.hpp"
#include "./layout_transforms.hpp"
#include "./macros.hpp"
#include "./mapped_functions.hpp"
#include "./mem/policies.hpp"
#include "./stdutil/array.hpp"

#include <algorithm>
#include <concepts>
#include <iostream>
#include <ranges>
#include <type_traits>

namespace nda {

  /**
   * @addtogroup av_factories
   * @{
   */

  /**
   * @brief Create an identity nda::matrix with ones on the diagonal.
   *
   * @tparam S nda::Scalar value type of the matrix.
   * @tparam Int Integral type.
   * @param dim Dimension of the square matrix.
   * @return Identity nda::matrix of size `dim x dim`.
   */
  template <Scalar S, std::integral Int = long>
  auto eye(Int dim) {
    auto r = matrix<S>(dim, dim);
    r      = S{1};
    return r;
  }

  /**
   * @ingroup av_math
   * @brief Get the trace of a 2-dimensional square array/view.
   *
   * @tparam M nda::ArrayOfRank<2> type.
   * @param m 2-dimensional array/view.
   * @return Sum of the diagonal elements of the array/view.
   */
  template <ArrayOfRank<2> M>
  auto trace(M const &m) {
    static_assert(get_rank<M> == 2, "Error in nda::trace: Array/View must have rank 2");
    EXPECTS(m.shape()[0] == m.shape()[1]);
    auto r = get_value_t<M>{};
    auto d = m.shape()[0];
    for (int i = 0; i < d; ++i) r += m(i, i);
    return r;
  }

  /**
   * @ingroup av_math
   * @brief Get the conjugate transpose of 2-dimensional array/view.
   *
   * @details It first calls nda::transpose and then the lazy nda::conj function in case the array/view is complex
   * valued.
   *
   * @tparam M nda::ArrayOfRank<2> type.
   * @param m 2-dimensional array/view.
   * @return (Lazy) Conjugate transpose of the array/view.
   */
  template <ArrayOfRank<2> M>
  ArrayOfRank<2> auto dagger(M const &m) {
    if constexpr (is_complex_v<typename M::value_type>)
      return conj(transpose(m));
    else
      return transpose(m);
  }

  /**
   * @brief Get a view of the diagonal of a 2-dimensional array/view.
   *
   * @tparam M nda::MemoryArrayOfRank<2> type.
   * @param m 2-dimensional array/view.
   * @return A view with the 'V' algebra of the diagonal of the array/view.
   */
  template <MemoryArrayOfRank<2> M>
  ArrayOfRank<1> auto diagonal(M &&m) { // NOLINT
    long dim    = std::min(m.shape()[0], m.shape()[1]);
    long stride = stdutil::sum(m.indexmap().strides());
    using vector_view_t =
       basic_array_view<std::remove_reference_t<decltype(*m.data())>, 1, C_stride_layout, 'V', nda::default_accessor, nda::borrowed<>>;
    return vector_view_t{C_stride_layout::mapping<1>{{dim}, {stride}}, m.data()};
  }

  /**
   * @brief Get a new nda::matrix with the given values on the diagonal.
   *
   * @tparam V nda::ArrayOfRank<1> or `std::ranges::contiguous_range` type.
   * @param v 1-dimensional array/view/container containing the diagonal values.
   * @return nda::matrix with the given values on the diagonal.
   */
  template <typename V>
    requires(std::ranges::contiguous_range<V> or ArrayOfRank<V, 1>)
  ArrayOfRank<2> auto diag(V const &v) {
    if constexpr (std::ranges::contiguous_range<V>) {
      return diag(nda::basic_array_view{v});
    } else {
      auto m      = matrix<std::remove_const_t<typename V::value_type>>::zeros({v.size(), v.size()});
      diagonal(m) = v;
      return m;
    }
  }

  /**
   * @brief Stack two 2-dimensional arrays/views vertically.
   *
   * @details This is a more restricted implementation then nda::concatenate. It is only for 2D arrays/views.
   *
   * Given a an array A of size `n x q` and an array B of size `p x q`, the function returns a new array C of size
   * `(n + p) x q` such that `C(range(0, n), range::all) == A` and `C(range(n, n + p), range::all) == B` is true.
   *
   * @tparam A nda::ArrayOfRank<2> type.
   * @tparam B nda::ArrayOfRank<2> type.
   * @param a 2-dimensional array/view.
   * @param b 2-dimensional array/view.
   * @return A new 2-dimensional array/view with the two arrays stacked vertically.
   */
  template <ArrayOfRank<2> A, ArrayOfRank<2> B>
    requires(std::same_as<get_value_t<A>, get_value_t<B>>) // NB the get_value_t gets rid of const if any
  matrix<get_value_t<A>> vstack(A const &a, B const &b) {
    static_assert(get_rank<A> == 2, "Error in nda::vstack: Only rank 2 arrays/views are allowed");
    static_assert(get_rank<A> == 2, "Error in nda::vstack: Only rank 2 arrays/views are allowed");
    EXPECTS_WITH_MESSAGE(a.shape()[1] == b.shape()[1], "Error in nda::vstack: The second dimension of the two matrices must be equal");

    auto [n, q] = a.shape();
    auto p      = b.shape()[0];
    matrix<get_value_t<A>> res(n + p, q);
    res(range(0, n), range::all)     = a;
    res(range(n, n + p), range::all) = b;
    return res;
  }

  /**
   * @brief Check if a given matrix is square, i.e. if the first dimension has the same extent as the second
   * dimension.
   *
   * @tparam A nda::Matrix type.
   * @param a Matrix to be checked.
   * @param print_error If true, print an error message if the matrix is not square.
   * @return True if the matrix is square, false otherwise.
   */
  template <Matrix A>
  bool is_matrix_square(A const &a, bool print_error = false) {
    auto const [m, n] = a.shape();
    if (m != n and print_error) std::cerr << "Error in nda::is_matrix_square: Dimensions are: (" << m << "," << n << ")\n" << std::endl;
    return m == n;
  }

  /**
   * @brief Check if a given matrix is diagonal, i.e. if it is square (see nda::is_matrix_square) and all the the
   * off-diagonal elements are zero.
   *
   * @tparam A nda::Matrix type.
   * @param a Matrix to be checked.
   * @param print_error If true, print an error message if the matrix is not diagonal.
   * @return True if the matrix is diagonal, false otherwise.
   */
  template <Matrix A>
  bool is_matrix_diagonal(A const &a, bool print_error = false) {
    bool const r = is_matrix_square(a) and a == diag(diagonal(a));
    if (not r and print_error) std::cerr << "Error in nda::is_matrix_diagonal: Non-diagonal matrix: " << a << std::endl;
    return r;
  }

  /** @} */

} // namespace nda
