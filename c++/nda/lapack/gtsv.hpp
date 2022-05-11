// Copyright (c) 2021 Simons Foundation
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

#pragma once

#include "../lapack.hpp"

namespace nda::lapack {

  /**
   * Solves the equation
   *
   *     A*X = B,
   *
   *  where A is an N-by-N tridiagonal matrix, by Gaussian elimination with
   *  partial pivoting.
   *
   *  Note that the equation  A**T *X = B  may be solved by interchanging the
   *  order of the arguments du and dl.
   *
   * [in,out]  dl is real/complex array, dimension (N-1)
   *           On entry, dl must contain the (n-1) subdiagonal elements of
   *           A.
   *           On exit, dl is overwritten by the (n-2) elements of the
   *           second superdiagonal of the upper triangular matrix U from
   *           the LU factorization of A, in dl(1), ..., dl(n-2).
   *
   * [in,out]  d is real/complex array, dimension (N)
   *           On entry, D must contain the diagonal elements of A.
   *           On exit, D is overwritten by the n diagonal elements of U.
   *
   * [in,out]  du is real/complex array, dimension (N-1)
   *           On entry, du must contain the (n-1) superdiagonal elements
   *           of A.
   *           On exit, du is overwritten by the (n-1) elements of the first
   *           superdiagonal of U.
   *
   * [in,out]  b is real/complex array, dimension (LDB,NRHS)
   *           On entry, the N-by-NRHS right hand side matrix B.
   *           On exit, if INFO = 0, the N-by-NRHS solution matrix X.
   *
   * [return]  INFO is INTEGER
   *           = 0:  successful exit
   *           < 0:  if INFO = -i, the i-th argument had an illegal value
   *           > 0:  if INFO = i, U(i,i) is exactly zero, and the solution
   *                 has not been computed.  The factorization has not been
   *                 completed unless i = N.
   */
  template <MemoryVector DL, MemoryVector D, MemoryVector DU, MemoryArray B>
  requires(have_same_value_type_v<DL, D, DU, B> and mem::on_host<DL, D, DU, B> and is_blas_lapack_v<get_value_t<DL>>)
  int gtsv(DL &dl, D &d, DU &du, B &b) {
    static_assert((get_rank<B> == 1 or get_rank<B> == 2), "gtsv: M must be an matrix/array/view of rank  1 or 2");

    int N    = d.extent(0);
    int NRHS = (get_rank<B> == 2 ? b.extent(1) : 1);
    EXPECTS(dl.extent(0) == d.extent(0) - 1); // "gtsv : dimension mismatch between sub-diagonal and diagonal vectors "
    EXPECTS(du.extent(0) == d.extent(0) - 1); // "gtsv : dimension mismatch between super-diagonal and diagonal vectors "
    EXPECTS(b.extent(0) == d.extent(0));      // "gtsv : dimension mismatch between diagonal vector and RHS matrix, "

    int info = 0;
    f77::gtsv(N, NRHS, dl.data(), d.data(), du.data(), b.data(), N, info);
    return info;
  }
} // namespace nda::lapack
