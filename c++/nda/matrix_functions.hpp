// Copyright (c) 2019-2020 Simons Foundation
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

#pragma once

namespace nda {

  template <typename V>
  matrix<V> eye(long dim) {
    matrix<V> r(dim, dim);
    r = 1;
    return r;
  }

  template <ArrayOfRank<2> M>
  auto trace(M const &m)  {
    static_assert(get_rank<M> == 2, "trace: array must have rank two");
    EXPECTS(m.shape()[0] == m.shape()[1]);
    auto r = get_value_t<M>{};
    auto d = m.shape()[0];
    for (int i = 0; i < d; ++i) r += m(i, i);
    return r;
  }

  ///
  template <ArrayOfRank<2> M>
  Array auto dagger(M const &m)  {
    static_assert(get_rank<M> == 2, "dagger: array must have rank two");
    if constexpr (is_complex_v<typename M::value_type>)
      return conj(transpose(m));
    else
      return transpose(m);
  }

  ///
  /// Give 2 matrices A (of size n x q) and B (of size p x q)
  /// produces a new matrix of size
  ///
  template <ArrayOfRank<2> A, ArrayOfRank<2> B>
  REQUIRES20(std::same_as<get_value_t<A>, get_value_t<B>>) // NB the get_value_t gets rid of const if any
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
