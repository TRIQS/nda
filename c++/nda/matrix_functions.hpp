// Copyright (c) 2019-2021 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once

#include "basic_array.hpp"
#include "declarations.hpp"

namespace nda {

  /// Create a two-dimensional matrix of scalar-type T with ones on the diagonal and zeros elsewhere
  template <Scalar S, std::integral Int = long>
  auto eye(Int dim) {
    auto r = matrix<S>(dim, dim);
    r = S{1};
    return r;
  }

  /// Return the trace of a matrix or a rank==2 array
  template <ArrayOfRank<2> M>
  auto trace(M const &m)  {
    static_assert(get_rank<M> == 2, "trace: array must have rank two");
    EXPECTS(m.shape()[0] == m.shape()[1]);
    auto r = get_value_t<M>{};
    auto d = m.shape()[0];
    for (int i = 0; i < d; ++i) r += m(i, i);
    return r;
  }

  /// Return the conjugate transpase of a matrix or a rank==2 array
  template <ArrayOfRank<2> M>
  ArrayOfRank<2> auto dagger(M const &m)  {
    if constexpr (is_complex_v<typename M::value_type>)
      return conj(transpose(m));
    else
      return transpose(m);
  }

  /// Return a vector_view on the diagonal of a matrix or a rank==2 array
  template <MemoryArrayOfRank<2> M>
  ArrayOfRank<1> auto diagonal(M &m) {
    long dim = std::min(m.shape()[0], m.shape()[1]);
    long stride = stdutil::sum(m.indexmap().strides());
    using vector_view_t =
       basic_array_view<std::remove_reference_t<decltype(*m.data())>, 1, C_stride_layout, 'V', nda::default_accessor, nda::borrowed<>>;
    return vector_view_t{C_stride_layout::mapping<1>{{dim}, {stride}}, m.data()};
  }

  /// Return a new matrix with the values of v on the diagonal
  template <ArrayOfRank<1> V>
  ArrayOfRank<2> auto diag(V const &v)  {
    auto m      = matrix<std::remove_const_t<typename V::value_type>>::zeros({v.size(), v.size()});
    diagonal(m) = v;
    return m;
  }

  /// Give 2 matrices A (of size n x q) and B (of size p x q)
  /// produces a new matrix C of size (n + p) x q such that
  /// C[0:n,:] == A and C[n:n+p,:] == B
  template <ArrayOfRank<2> A, ArrayOfRank<2> B>
  requires(std::same_as<get_value_t<A>, get_value_t<B>>) // NB the get_value_t gets rid of const if any
  matrix<get_value_t<A>> vstack(A const &a, B const &b)
      {
    static_assert(get_rank<A> == 2, "vstack: first argument must have rank two");
    static_assert(get_rank<B> == 2, "vstack: second argument must have rank two");
    EXPECTS_WITH_MESSAGE(a.shape()[1] == b.shape()[1],
                         "vstack. The second dimension of the two matrices must be equal but \n   a is of shape " + to_string(a.shape())
                            + "   b is of shape" + to_string(b.shape()));
    // Impl. Concept only ! A, B can be expression template, e.g.

    auto [n, q] = a.shape();
    auto p      = b.shape()[0];
    auto _      = range::all;

    matrix<get_value_t<A>> res(n + p, q);
    res(range(0, n), _)     = a;
    res(range(n, n + p), _) = b;
    return res;
  }

} // namespace nda
