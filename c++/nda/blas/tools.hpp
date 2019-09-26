#pragma once
#include <complex>
#include <type_traits>

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
  template <typename A0, typename... A>
  inline constexpr bool have_same_element_type_and_it_is_blas_type_v =
     is_blas_lapack_v<typename A0::value_type> and (std::is_same_v<typename A::value_type, typename A0::value_type> and ... and true);

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
