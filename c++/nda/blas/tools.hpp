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

namespace nda {
  using dcomplex = std::complex<double>;
}

namespace nda::blas {

  // ================================================

  // FIXME : move to impl NS
  template <typename MatrixType>
  char get_trans(MatrixType const &A, bool transpose) {
    return (A.indexmap().is_stride_order_Fortran() ? (transpose ? 'T' : 'N') : (transpose ? 'N' : 'T'));
  }

  // returns the # of rows of the matrix *seen* as fortran matrix
  template <typename MatrixType>
  size_t get_n_rows(MatrixType const &A) {
    return (A.indexmap().is_stride_order_Fortran() ? A.extent(0) : A.extent(1));
  }

  // returns the # of cols of the matrix *seen* as fortran matrix
  template <typename MatrixType>
  size_t get_n_cols(MatrixType const &A) {
    return (A.indexmap().is_stride_order_Fortran() ? A.extent(1) : A.extent(0));
  }

  // LDA in lapack jargon
  template <typename MatrixType>
  int get_ld(MatrixType const &A) {
    return A.indexmap().strides()[A.indexmap().is_stride_order_Fortran() ? 1 : 0];
  }

  //template <typename M>
  //bool min_stride_is_1(M const &m) {
  //return a.indexmap().min_stride() == 1;
  //}v

} // namespace nda::blas
