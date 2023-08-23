// Copyright (c) 2022-2023 Simons Foundation
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
// Authors: Miguel Morales, Nils Wentzell

#include <nda/nda.hpp>
#include <nda/device.hpp>
#include "cublas_interface.hpp"

#ifdef NDA_HAVE_MAGMA
#include "magma_v2.h"
#endif

#include <string>
#include <vector>

using namespace std::string_literals;

namespace nda::blas::device {

  // Local function to get unique CuBlas Handle, Used by all routines
  inline cublasHandle_t &get_handle() {
    struct handle_storage_t { // RAII for handle
      handle_storage_t() { cublasCreate(&handle); }
      ~handle_storage_t() { cublasDestroy(handle); }
      cublasHandle_t handle = {};
    };
    static auto sto = handle_storage_t{};
    return sto.handle;
  }

#ifdef NDA_HAVE_MAGMA
  constexpr magma_trans_t get_magma_op(char op) {
    switch (op) {
      case 'N': return MagmaNoTrans; break;
      case 'T': return MagmaTrans; break;
      case 'C': return MagmaConjTrans; break;
      default: std::terminate(); return {};
    }
  }

  // Get Magma queue, Used by all magma routines
  magma_queue_t &get_magma_queue() {
    struct queue_t {
      queue_t() {
        int device{};
        magma_getdevice(&device);
        magma_queue_create(device, &q);
      }
      ~queue_t() { magma_queue_destroy(q); }
      operator magma_queue_t() { return q; }

      private:
      magma_queue_t q = {};
    };
    static queue_t q = {};
    return q;
  }
#endif

  /// Global option to turn on/off the cudaDeviceSynchronize after cublas library calls
  static bool synchronize = true;
#define CUBLAS_CHECK(X, ...)                                                                                                                         \
  {                                                                                                                                                  \
    auto err = X(get_handle(), __VA_ARGS__);                                                                                                         \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                                                                              \
      NDA_RUNTIME_ERROR << AS_STRING(X) << " failed \n"                                                                                              \
                        << " cublasGetStatusName: " << cublasGetStatusName(err) << "\n"                                                              \
                        << " cublasGetStatusString: " << cublasGetStatusString(err) << "\n";                                                         \
    }                                                                                                                                                \
    if (synchronize) {                                                                                                                               \
      auto errsync = cudaDeviceSynchronize();                                                                                                        \
      if (errsync != cudaSuccess) {                                                                                                                      \
        NDA_RUNTIME_ERROR << " cudaDeviceSynchronize failed after call to: " << AS_STRING(X) << "\n"                                                 \
                          << " cudaGetErrorName: " << cudaGetErrorName(errsync) << "\n"                                                              \
                          << " cudaGetErrorString: " << cudaGetErrorString(errsync) << "\n";                                                         \
      }                                                                                                                                              \
    }                                                                                                                                                \
  }

  void gemm(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC) {
    CUBLAS_CHECK(cublasDgemm, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha, A, LDA, B, LDB, &beta, C, LDC);
  }
  void gemm(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex *A, int LDA, const dcomplex *B, int LDB, dcomplex beta,
            dcomplex *C, int LDC) {
    auto alpha_cu = cucplx(alpha);
    auto beta_cu  = cucplx(beta);
    CUBLAS_CHECK(cublasZgemm, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha_cu, cucplx(A), LDA, cucplx(B), LDB, &beta_cu, cucplx(C), LDC);
  }

  void gemm_batch(char op_a, char op_b, int M, int N, int K, double alpha, const double **A, int LDA, const double **B, int LDB, double beta,
                  double **C, int LDC, int batch_count) {
    CUBLAS_CHECK(cublasDgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha, A, LDA, B, LDB, &beta, C, LDC, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex **A, int LDA, const dcomplex **B, int LDB, dcomplex beta,
                  dcomplex **C, int LDC, int batch_count) {
    auto alpha_cu = cucplx(alpha);
    auto beta_cu  = cucplx(beta);
    CUBLAS_CHECK(cublasZgemmBatched, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha_cu, cucplx(A), LDA, cucplx(B), LDB, &beta_cu,
                 cucplx(C), LDC, batch_count);
  }

#ifdef NDA_HAVE_MAGMA
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, double alpha, const double **A, int *LDA, const double **B, int *LDB, double beta,
                   double **C, int *LDC, int batch_count) {
    magmablas_dgemm_vbatched(get_magma_op(op_a), get_magma_op(op_b), M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC, batch_count, get_magma_queue());
    if (synchronize) magma_queue_sync(get_magma_queue());
    if (synchronize) cudaDeviceSynchronize();
  }
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, dcomplex alpha, const dcomplex **A, int *LDA, const dcomplex **B, int *LDB,
                   dcomplex beta, dcomplex **C, int *LDC, int batch_count) {
    auto alpha_cu = cucplx(alpha);
    auto beta_cu  = cucplx(beta);
    magmablas_zgemm_vbatched(get_magma_op(op_a), get_magma_op(op_b), M, N, K, alpha_cu, cucplx(A), LDA, cucplx(B), LDB, beta_cu, cucplx(C), LDC,
                             batch_count, get_magma_queue());
    if (synchronize) magma_queue_sync(get_magma_queue());
    if (synchronize) cudaDeviceSynchronize();
  }
#endif

  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, int strideA, const double *B, int LDB,
                          int strideB, double beta, double *C, int LDC, int strideC, int batch_count) {
    CUBLAS_CHECK(cublasDgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha, A, LDA, strideA, B, LDB, strideB, &beta, C,
                 LDC, strideC, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex *A, int LDA, int strideA, const dcomplex *B,
                          int LDB, int strideB, dcomplex beta, dcomplex *C, int LDC, int strideC, int batch_count) {
    auto alpha_cu = cucplx(alpha);
    auto beta_cu  = cucplx(beta);
    CUBLAS_CHECK(cublasZgemmStridedBatched, get_cublas_op(op_a), get_cublas_op(op_b), M, N, K, &alpha_cu, cucplx(A), LDA, strideA, cucplx(B), LDB,
                 strideB, &beta_cu, cucplx(C), LDC, strideC, batch_count);
  }

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy) { cublasDaxpy(get_handle(), N, &alpha, x, incx, Y, incy); }
  void axpy(int N, dcomplex alpha, const dcomplex *x, int incx, dcomplex *Y, int incy) {
    CUBLAS_CHECK(cublasZaxpy, N, cucplx(&alpha), cucplx(x), incx, cucplx(Y), incy);
  }

  void copy(int N, const double *x, int incx, double *Y, int incy) { cublasDcopy(get_handle(), N, x, incx, Y, incy); }
  void copy(int N, const dcomplex *x, int incx, dcomplex *Y, int incy) { CUBLAS_CHECK(cublasZcopy, N, cucplx(x), incx, cucplx(Y), incy); }

  double dot(int M, const double *x, int incx, const double *Y, int incy) {
    double res{};
    CUBLAS_CHECK(cublasDdot, M, x, incx, Y, incy, &res);
    return res;
  }
  dcomplex dot(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy) {
    cuDoubleComplex res;
    CUBLAS_CHECK(cublasZdotu, M, cucplx(x), incx, cucplx(Y), incy, &res);
    return {res.x, res.y};
  }
  dcomplex dotc(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy) {
    cuDoubleComplex res;
    CUBLAS_CHECK(cublasZdotc, M, cucplx(x), incx, cucplx(Y), incy, &res);
    return {res.x, res.y};
  }

  void gemv(char op, int M, int N, double alpha, const double *A, int LDA, const double *x, int incx, double beta, double *Y, int incy) {
    CUBLAS_CHECK(cublasDgemv, get_cublas_op(op), M, N, &alpha, A, LDA, x, incx, &beta, Y, incy);
  }
  void gemv(char op, int M, int N, dcomplex alpha, const dcomplex *A, int LDA, const dcomplex *x, int incx, dcomplex beta, dcomplex *Y, int incy) {
    CUBLAS_CHECK(cublasZgemv, get_cublas_op(op), M, N, cucplx(&alpha), cucplx(A), LDA, cucplx(x), incx, cucplx(&beta), cucplx(Y), incy);
  }

  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA) {
    CUBLAS_CHECK(cublasDger, M, N, &alpha, x, incx, Y, incy, A, LDA);
  }
  void ger(int M, int N, dcomplex alpha, const dcomplex *x, int incx, const dcomplex *Y, int incy, dcomplex *A, int LDA) {
    CUBLAS_CHECK(cublasZgeru, M, N, cucplx(&alpha), cucplx(x), incx, cucplx(Y), incy, cucplx(A), LDA);
  }

  void scal(int M, double alpha, double *x, int incx) { CUBLAS_CHECK(cublasDscal, M, &alpha, x, incx); }
  void scal(int M, dcomplex alpha, dcomplex *x, int incx) { CUBLAS_CHECK(cublasZscal, M, cucplx(&alpha), cucplx(x), incx); }

  void swap(int N, double *x, int incx, double *Y, int incy) { CUBLAS_CHECK(cublasDswap, N, x, incx, Y, incy); }
  void swap(int N, dcomplex *x, int incx, dcomplex *Y, int incy) { CUBLAS_CHECK(cublasZswap, N, cucplx(x), incx, cucplx(Y), incy); }

} // namespace nda::blas::device
