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
#include <complex>
#include <type_traits>

#include "../traits.hpp"
#include "../concepts.hpp"

namespace nda {
  using dcomplex = std::complex<double>;
}

namespace nda::blas {

  // ================================================

  template <MemoryMatrix M>
  static constexpr bool has_F_layout = std::remove_cvref_t<M>::is_stride_order_Fortran();

  template <MemoryMatrix M>
  static constexpr bool has_C_layout = std::remove_cvref_t<M>::is_stride_order_C();

  // FIXME : move to impl NS
  template <typename MatrixType>
  char get_op(MatrixType const &A, bool transpose) {
    return (A.indexmap().is_stride_order_Fortran() ? (transpose ? 'T' : 'N') : (transpose ? 'N' : 'T'));
  }


  // LDA in lapack jargon
  template <MemoryMatrix A>
  int get_ld(A const &a) {
    return a.indexmap().strides()[has_F_layout<A> ? 1 : 0];
  }

  //template <typename M>
  //bool min_stride_is_1(M const &m) {
  //return a.indexmap().min_stride() == 1;
  //}v

} // namespace nda::blas
