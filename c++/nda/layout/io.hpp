// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2020 Simons Foundation
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
#include <ostream>

namespace std {

  template <typename T, size_t N>
  std::ostream &operator<<(std::ostream &out, std::array<T, N> const &v) {
    out << "(";
    for (size_t i = 0; i < N; ++i) out << (i == 0 ? "" : " ") << v[i];
    out << ")";
    return out;
  }

  /* template<typename T, size_t N> 
  std::string to_string(std::array<T,N> const &v) const {
    std::ostringstream fs;
    fs<<v;
    return fs.str();
  }
 */
} // namespace std

namespace nda {

  /* // stride_order_t
  template <int Rank> std::ostream &operator<<(std::ostream &out, stride_order_t<Rank> const &s) {
    out << "(";
    for (int i = 0; i < s.rank(); ++i) out << (i == 0 ? "" : " ") << int(s.as_permutation()[i]);
    out << ")";
    return out;
  }
*/
  /*
  namespace permutations {

    inline std::to_string(uint64_t perm) {
      std::stringstream fs;
      int s = size_of_permutation(perm);
      fs << "[" << apply(perm, 0);
      for (int i = 1; i < s; ++i) fs << ',' << apply(perm, i);
      fs << "]";
      return fs.str();
    }

  } // namespace permutations
*/
} // namespace nda
