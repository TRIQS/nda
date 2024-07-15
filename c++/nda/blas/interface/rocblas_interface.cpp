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
// Authors: Miguel Morales, Nils Wentzell, Geraud Krawezik

#include <nda/nda.hpp>
#include <nda/device.hpp>
#include "rocblas_interface.hpp"

#ifdef NDA_HAVE_MAGMA
#include "magma_v2.h"
#endif

#include "rocblas/rocblas.h"
#include "rocblas/internal/rocblas-complex-types.h"

#include <string>
#include <vector>

using namespace std::string_literals;

namespace nda::blas::device {

  // Local function to get unique rocBlas Handle, Used by all routines
  inline rocblas_handle &get_handle() {
    struct handle_storage_t { // RAII for handle
      handle_storage_t() { rocblas_create_handle(&handle); }
      ~handle_storage_t() { rocblas_destroy_handle(handle); }
      rocblas_handle handle = {};
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

  /// Global option to turn on/off the hipDeviceSynchronize after rocblas library calls
  static bool synchronize = true;
#define ROCBLAS_CHECK(X, ...)                                                                                                                        \
  {                                                                                                                                                  \
    auto err = X(get_handle(), __VA_ARGS__);                                                                                                         \
    if (err != rocblas_status_success) {                                                                                                             \
      NDA_RUNTIME_ERROR << AS_STRING(X) << " failed \n"                                                                                              \
                        << " rocblas_status: " << err << "\n";                                                                                       \
    }                                                                                                                                                \
    if (synchronize) {                                                                                                                               \
      auto errsync = hipDeviceSynchronize();                                                                                                         \
      if (errsync != hipSuccess) {                                                                                                                   \
        NDA_RUNTIME_ERROR << " hipDeviceSynchronize failed after call to: " << AS_STRING(X) << "\n"                                                  \
                          << " hipGetErrorName: " << hipGetErrorName(errsync) << "\n"                                                                \
                          << " hipGetErrorString: " << hipGetErrorString(errsync) << "\n";                                                           \
      }                                                                                                                                              \
    }                                                                                                                                                \
  }

  void gemm(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, const double *B, int LDB, double beta, double *C,
            int LDC) {
    ROCBLAS_CHECK(rocblas_dgemm, get_rocblas_op(op_a), get_rocblas_op(op_b), M, N, K, &alpha, A, LDA, B, LDB, &beta, C, LDC);
  }
  void gemm(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex *A, int LDA, const dcomplex *B, int LDB, dcomplex beta,
            dcomplex *C, int LDC) {
    auto alpha_roc = roccplx(alpha);
    auto beta_roc  = roccplx(beta);
    ROCBLAS_CHECK(rocblas_zgemm, get_rocblas_op(op_a), get_rocblas_op(op_b), M, N, K, &alpha_roc, roccplx(A), LDA, roccplx(B), LDB, &beta_roc,
                  roccplx(C), LDC);
  }

  void gemm_batch(char op_a, char op_b, int M, int N, int K, double alpha, const double **A, int LDA, const double **B, int LDB, double beta,
                  double **C, int LDC, int batch_count) {
    ROCBLAS_CHECK(rocblas_dgemm_batched, get_rocblas_op(op_a), get_rocblas_op(op_b), M, N, K, &alpha, A, LDA, B, LDB, &beta, C, LDC, batch_count);
  }
  void gemm_batch(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex **A, int LDA, const dcomplex **B, int LDB, dcomplex beta,
                  dcomplex **C, int LDC, int batch_count) {
    auto alpha_roc = roccplx(alpha);
    auto beta_roc  = roccplx(beta);
    ROCBLAS_CHECK(rocblas_zgemm_batched, get_rocblas_op(op_a), get_rocblas_op(op_b), M, N, K, &alpha_roc, roccplx(A), LDA, roccplx(B), LDB, &beta_roc,
                  roccplx(C), LDC, batch_count);
  }

#ifdef NDA_HAVE_MAGMA
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, double alpha, const double **A, int *LDA, const double **B, int *LDB, double beta,
                   double **C, int *LDC, int batch_count) {
    magmablas_dgemm_vbatched(get_magma_op(op_a), get_magma_op(op_b), M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC, batch_count, get_magma_queue());
    if (synchronize) magma_queue_sync(get_magma_queue());
    if (synchronize) hip_device_synchronize();
  }
  void gemm_vbatch(char op_a, char op_b, int *M, int *N, int *K, dcomplex alpha, const dcomplex **A, int *LDA, const dcomplex **B, int *LDB,
                   dcomplex beta, dcomplex **C, int *LDC, int batch_count) {
    auto alpha_roc = roccplx(alpha);
    auto beta_roc  = roccplx(beta);
    magmablas_zgemm_vbatched(get_magma_op(op_a), get_magma_op(op_b), M, N, K, alpha_roc, roccplx(A), LDA, roccplx(B), LDB, beta_roc, roccplx(C), LDC,
                             batch_count, get_magma_queue());
    if (synchronize) magma_queue_sync(get_magma_queue());
    if (synchronize) hip_device_synchronize();
  }
#endif

  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, double alpha, const double *A, int LDA, int strideA, const double *B, int LDB,
                          int strideB, double beta, double *C, int LDC, int strideC, int batch_count) {
    ROCBLAS_CHECK(rocblas_dgemm_strided_batched, get_rocblas_op(op_a), get_rocblas_op(op_b), M, N, K, &alpha, A, LDA, strideA, B, LDB, strideB, &beta,
                  C, LDC, strideC, batch_count);
  }
  void gemm_batch_strided(char op_a, char op_b, int M, int N, int K, dcomplex alpha, const dcomplex *A, int LDA, int strideA, const dcomplex *B,
                          int LDB, int strideB, dcomplex beta, dcomplex *C, int LDC, int strideC, int batch_count) {
    auto alpha_roc = roccplx(alpha);
    auto beta_roc  = roccplx(beta);
    ROCBLAS_CHECK(rocblas_zgemm_strided_batched, get_rocblas_op(op_a), get_rocblas_op(op_b), M, N, K, &alpha_roc, roccplx(A), LDA, strideA,
                  roccplx(B), LDB, strideB, &beta_roc, roccplx(C), LDC, strideC, batch_count);
  }

  void axpy(int N, double alpha, const double *x, int incx, double *Y, int incy) { rocblas_daxpy(get_handle(), N, &alpha, x, incx, Y, incy); }
  void axpy(int N, dcomplex alpha, const dcomplex *x, int incx, dcomplex *Y, int incy) {
    ROCBLAS_CHECK(rocblas_zaxpy, N, roccplx(&alpha), roccplx(x), incx, roccplx(Y), incy);
  }

  void copy(int N, const double *x, int incx, double *Y, int incy) { rocblas_dcopy(get_handle(), N, x, incx, Y, incy); }
  void copy(int N, const dcomplex *x, int incx, dcomplex *Y, int incy) { ROCBLAS_CHECK(rocblas_zcopy, N, roccplx(x), incx, roccplx(Y), incy); }

  double dot(int M, const double *x, int incx, const double *Y, int incy) {
    double res{};
    ROCBLAS_CHECK(rocblas_ddot, M, x, incx, Y, incy, &res);
    return res;
  }
  dcomplex dot(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy) {
    rocblas_double_complex res;
    ROCBLAS_CHECK(rocblas_zdotu, M, roccplx(x), incx, roccplx(Y), incy, &res);
    return {res.x, res.y};
  }
  dcomplex dotc(int M, const dcomplex *x, int incx, const dcomplex *Y, int incy) {
    rocblas_double_complex res;
    ROCBLAS_CHECK(rocblas_zdotc, M, roccplx(x), incx, roccplx(Y), incy, &res);
    return {res.x, res.y};
  }

  void gemv(char op, int M, int N, double alpha, const double *A, int LDA, const double *x, int incx, double beta, double *Y, int incy) {
    ROCBLAS_CHECK(rocblas_dgemv, get_rocblas_op(op), M, N, &alpha, A, LDA, x, incx, &beta, Y, incy);
  }
  void gemv(char op, int M, int N, dcomplex alpha, const dcomplex *A, int LDA, const dcomplex *x, int incx, dcomplex beta, dcomplex *Y, int incy) {
    ROCBLAS_CHECK(rocblas_zgemv, get_rocblas_op(op), M, N, roccplx(&alpha), roccplx(A), LDA, roccplx(x), incx, roccplx(&beta), roccplx(Y), incy);
  }

  void ger(int M, int N, double alpha, const double *x, int incx, const double *Y, int incy, double *A, int LDA) {
    ROCBLAS_CHECK(rocblas_dger, M, N, &alpha, x, incx, Y, incy, A, LDA);
  }
  void ger(int M, int N, dcomplex alpha, const dcomplex *x, int incx, const dcomplex *Y, int incy, dcomplex *A, int LDA) {
    ROCBLAS_CHECK(rocblas_zgeru, M, N, roccplx(&alpha), roccplx(x), incx, roccplx(Y), incy, roccplx(A), LDA);
  }

  void scal(int M, double alpha, double *x, int incx) { ROCBLAS_CHECK(rocblas_dscal, M, &alpha, x, incx); }
  void scal(int M, dcomplex alpha, dcomplex *x, int incx) { ROCBLAS_CHECK(rocblas_zscal, M, roccplx(&alpha), roccplx(x), incx); }

  void swap(int N, double *x, int incx, double *Y, int incy) { ROCBLAS_CHECK(rocblas_dswap, N, x, incx, Y, incy); }
  void swap(int N, dcomplex *x, int incx, dcomplex *Y, int incy) { ROCBLAS_CHECK(rocblas_zswap, N, roccplx(x), incx, roccplx(Y), incy); }

} // namespace nda::blas::device
