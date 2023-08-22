// Copyright (c) 2023 Simons Foundation
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
// Authors: Nils Wentzell

#include "../basic_array.hpp"
#include "../concepts.hpp"
#include "../blas.hpp"

namespace nda {

  // FIXMEOP : for p=2 (and only there)
  // we make a full copy for an expression, just to call dotc ??

  /**
   * Calculate the p-norm of an array x with scalar values and rank one:
   *
   *   norm(x) = sum(abs(x)^ord)^(1./ord)
   *
   * with the special cases (following numpy.linalg.norm convention)
   *
   *   norm(x, 0.0)  = number of non-zero elements
   *   norm(x, inf)  = max_element(abs(x))
   *   norm(x, -inf) = min_element(abs(x))
   * 
   * @param x The array to calculate the norm of
   * @param p The order of the norm [default=2.0]
   * @tparam A The type of the array
   * @return The norm as a double
   */
  template <ArrayOfRank<1> A>
  double norm(A const &x, double p = 2.0) {
    // Scalar check: Can't move to template constraint, get_value_t not generically implemented
    static_assert(Scalar<get_value_t<A>>, "norm only works for arrays with scalar values");

    if (p == 2.0) [[likely]] {
      if constexpr (MemoryArray<A>)
        return std::sqrt(std::real(nda::blas::dotc(x, x)));
      else
        return norm(make_regular(x));
    } else if (p == 1.0) {
      return sum(abs(x));
    } else if (p == 0.0) {
      // return std::count_if(a.begin(), a.end(), [](S s) { return s != S{0}; }); Fails for nda::expr
      long count = 0;
      for (long i = 0; i < x.size(); ++i) {
        if (x(i) != get_value_t<A>{0}) ++count;
      }
      return double(count);
    } else if (p == std::numeric_limits<double>::infinity()) {
      return max_element(abs(x));
    } else if (p == -std::numeric_limits<double>::infinity()) {
      return min_element(abs(x));
    } else {
      return std::pow(sum(pow(abs(x), p)), 1.0 / p);
    }
  }
} // namespace nda
