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

  template <CONCEPT(ArrayOfRank<2>) M>
  auto trace(M const &m) REQUIRES17(is_ndarray_v<M>) {
    static_assert(get_rank<M> == 2);
    EXPECTS(m.shape()[0] == m.shape()[1]);
    auto r = get_value_t<M>{};
    auto d = m.shape()[0];
    for (int i = 0; i < d; ++i) r += m(i, i);
    return r;
  }

  ///
  template <CONCEPT(ArrayOfRank<2>) M>
  AUTO(Array)
  dagger(M const &m) REQUIRES17(is_ndarray_v<M> and (get_rank<M> == 2)) {
    if constexpr (is_complex_v<typename M::value_type>)
      return conj(transpose(m));
    else
      return transpose(m);
  }

  ///
  /// Give 2 matrices A (of size n x q) and B (of size p x q)
  /// produces a new matrix of size
  ///
  template <CONCEPT(ArrayOfRank<2>) A, CONCEPT(ArrayOfRank<2>) B>
  REQUIRES20(std::same_as<get_value_t<A>, get_value_t<B>>) // NB the get_value_t gets rid of const if any
  matrix<get_value_t<A>> vstack(A const &a, B const &b)
  //REQUIRES17(is_ndarray_v<A> and (get_rank<A> == 2) and is_ndarray_v<B> and (get_rank<B> == 2))
  {
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
