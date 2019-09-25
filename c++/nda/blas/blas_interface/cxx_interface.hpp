/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *   author : O. Parcollet, P. Dumitrescu, N. Wentzell
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#pragma once

#include <complex>

namespace nda::blas::f77 {
  // typedef std::complex<double> dcomplex;

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy);
  void axpy(int N, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy);

  void copy(int N, const double *x, int incx, double *Y, int incy);
  void copy(int N, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy);

  double dot(int M, const double *x, int incx, const double *Y, int incy);
  //std::complex<double> dot (int  M, const std::complex<double>* x, int  incx, const std::complex<double>* Y, int  incy) ;

  void gemm(char trans_a, char trans_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC);
  void gemm(char trans_a, char trans_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA,
            const std::complex<double> *B, int LDB, std::complex<double> beta, std::complex<double> *C, int LDC);

  void gemv(char trans, int M, int N, double &alpha, const double *A, int &LDA, const double *x, int incx, double &beta, double *Y, int incy);
  void gemv(char trans, int M, int N, std::complex<double> &alpha, const std::complex<double> *A, int &LDA, const std::complex<double> *x, int incx,
            std::complex<double> &beta, std::complex<double> *Y, int incy);

  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA);
  void ger(int M, int N, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy,
           std::complex<double> *A, int LDA);

  void scal(int M, double alpha, double *x, int incx);
  void scal(int M, std::complex<double> alpha, std::complex<double> *x, int incx);

  void swap(int N, double *x, int incx, double *Y, int incy);
  void swap(int N, std::complex<double> *x, int incx, std::complex<double> *Y, int incy);

} // namespace nda::blas::f77
