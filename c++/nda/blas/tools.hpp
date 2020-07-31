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
#include <complex>
#include <type_traits>

namespace nda {

  using dcomplex = std::complex<double>;

}

namespace nda::blas {

  // a trait to detect all types for which blas/lapack bindings is defined
  // at the moment double and std::complex<double>
  // We don't need simple precision for the moment... To be added ?
  template <typename T>
  struct _is_blas_lapack : std::false_type {};
  template <>
  struct _is_blas_lapack<double> : std::true_type {};
  template <>
  struct _is_blas_lapack<std::complex<double>> : std::true_type {};

  template <typename T>
  inline constexpr bool is_blas_lapack_v = _is_blas_lapack<std::remove_const_t<T>>::value;

  // check all A have the same element_type
  // remove the ref here, this trait is exposed in the doc, it is simpler
  template <typename A0, typename... A>
  inline constexpr bool have_same_value_type_v = (std::is_same_v<std::remove_const_t<typename std::remove_reference_t<A>::value_type>,
                                                                 std::remove_const_t<typename std::remove_reference_t<A0>::value_type>> and ...
                                                  and true);

#if __cplusplus > 201703L

  template <typename T>
  concept IsDoubleOrComplex = is_blas_lapack_v<T>;

  // anyway from which I can make a MatrixView out
  template <typename A>
  concept MatrixView = (is_regular_or_view_v<std::decay_t<A>> and get_rank<std::decay_t<A>> == 2);

  // anyway from which I can make a VectorView out
  template <typename A>
  concept VectorView = (is_regular_or_view_v<std::decay_t<A>> and get_rank<std::decay_t<A>> == 1);

#endif

  // FIXME : kill this
  template <typename A0, typename... A>
  inline constexpr bool have_same_element_type_and_it_is_blas_type_v = have_same_value_type_v<A0, A...> and is_blas_lapack_v<typename A0::value_type>;

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
