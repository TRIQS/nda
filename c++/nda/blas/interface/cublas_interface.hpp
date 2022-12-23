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

namespace nda::blas::device {

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy);
  void axpy(int N, std::complex<double> alpha, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy);

  void copy(int N, const double *x, int incx, double *Y, int incy);
  void copy(int N, const std::complex<double> *x, int incx, std::complex<double> *Y, int incy);

  double dot(int M, const double *x, int incx, const double *Y, int incy);
  std::complex<double> dot(int M, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy);
  std::complex<double> dotc(int M, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy);

  void gemm(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC);
  void gemm(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA,
            const std::complex<double> *B, int LDB, std::complex<double> beta, std::complex<double> *C, int LDC);

  void gemm_batch(char op_a, char op_b, int M, int N, int K, double alpha, const double **A, int LDA, const double **B, int LDB, double beta,
                  double **C, int LDC, int batch_count);
  void gemm_batch(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> **A, int LDA,
                  const std::complex<double> **B, int LDB, std::complex<double> beta, std::complex<double> **C, int LDC, int batch_count);

  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, double alpha, const double **A, int *LDA, const double **B, int *LDB, double beta,
                   double **C, int *LDC, int batch_count);
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, std::complex<double> alpha, const std::complex<double> **A, int *LDA,
                   const std::complex<double> **B, int *LDB, std::complex<double> beta, std::complex<double> **C, int *LDC, int batch_count);

  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, int strideA, const double *B, int LDB,
                          int strideB, double beta, double *C, int LDC, int strideC, int batch_count);
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, std::complex<double> alpha, const std::complex<double> *A, int LDA, int strideA,
                          const std::complex<double> *B, int LDB, int srideB, std::complex<double> beta, std::complex<double> *C, int LDC,
                          int strideC, int batch_count);

  void gemv(char op, int M, int N, double alpha, const double *A, int LDA, const double *x, int incx, double beta, double *Y, int incy);
  void gemv(char op, int M, int N, std::complex<double> alpha, const std::complex<double> *A, int LDA, const std::complex<double> *x, int incx,
            std::complex<double> beta, std::complex<double> *Y, int incy);

  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA);
  void ger(int M, int N, std::complex<double> alpha, const std::complex<double> *x, int incx, const std::complex<double> *Y, int incy,
           std::complex<double> *A, int LDA);

  void scal(int M, double alpha, double *x, int incx);
  void scal(int M, std::complex<double> alpha, std::complex<double> *x, int incx);

  void swap(int N, double *x, int incx, double *Y, int incy);
  void swap(int N, std::complex<double> *x, int incx, std::complex<double> *Y, int incy);

} // namespace nda::blas::device
