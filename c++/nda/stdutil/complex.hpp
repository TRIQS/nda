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
//
// Authors: Olivier Parcollet, Nils Wentzell

#ifndef STDUTILS_COMPLEX_H
#define STDUTILS_COMPLEX_H

#include <complex>
#include <concepts>
#include <type_traits>

using namespace std::literals::complex_literals;

namespace std { // has to be in the right namespace for ADL !

#define IMPL_OP(OP)                                                                                                    \
  template <typename T, typename U>                                                                                    \
    requires(std::is_arithmetic_v<T> and std::is_arithmetic_v<U> and std::common_with<T, U>)                           \
  auto operator OP(std::complex<T> const &x, U y) {                                                                    \
    using C = std::complex<std::common_type_t<T, U>>;                                                                  \
    return C(x.real(), x.imag()) OP C(y);                                                                              \
  }                                                                                                                    \
                                                                                                                       \
  template <typename T, typename U>                                                                                    \
    requires(std::is_arithmetic_v<T> and std::is_arithmetic_v<U> and std::common_with<T, U>)                           \
  auto operator OP(T x, std::complex<U> const &y) {                                                                    \
    using C = std::complex<std::common_type_t<T, U>>;                                                                  \
    return C(x) OP C(y.real(), y.imag());                                                                              \
  }                                                                                                                    \
                                                                                                                       \
  template <typename T, typename U>                                                                                    \
    requires(std::is_arithmetic_v<T> and std::is_arithmetic_v<U> and std::common_with<T, U> and !std::is_same_v<T, U>) \
  auto operator OP(std::complex<T> const &x, std::complex<U> const &y) {                                               \
    using C = std::complex<std::common_type_t<T, U>>;                                                                  \
    return C(x.real(), x.imag()) OP C(y.real(), y.imag());                                                             \
  }
  IMPL_OP(+)
  IMPL_OP(-)
  IMPL_OP(*)
  IMPL_OP(/)
#undef IMPL_OP
}
#endif
