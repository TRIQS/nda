// Copyright (c) 2020-2021 Simons Foundation
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

namespace nda::lapack {

  ///
  template <MatrixView A, MatrixView B>
  int gels(const char TRANS, A &a, B &b) requires(nda::blas::have_same_element_type_and_it_is_blas_type_v<A, B>) 
  {
    int info = 0;
    using T = typename A::value_type;
    if constexpr (std::is_same_v<T, double>) {
      if (not(TRANS=='N' or TRANS=='T')) 
	NDA_RUNTIME_ERROR << "Error in gels : Incorrect parameter TRANS = " << TRANS;
    } else if constexpr (std::is_same_v<T, dcomplex>) {
      if (not(TRANS=='N' or TRANS=='C')) 
	NDA_RUNTIME_ERROR << "Error in gels : Incorrect parameter TRANS = " << TRANS;
    } else
      static_assert(false and always_true<A>, "Internal logic error");

    // We enforce Fortran order on B by making a copy if necessary.
    if constexpr (not B::layout_t::is_stride_order_Fortran()) {

      auto bf = matrix<T, F_layout>{b};
      info    = gels(TRANS,a, bf);
      b = bf;	
      return info;

    } else { // do not compile useless code !

      // Must be lapack compatible
      EXPECTS(a.indexmap().min_stride() == 1);
      EXPECTS(b.indexmap().min_stride() == 1);

      int nrhs = get_n_cols(b);

      // first call to get the optimal lwork
      T work1[1];

      if constexpr (A::layout_t::is_stride_order_Fortran()) {
  
        f77::gels(TRANS,get_n_rows(a), get_n_cols(a), nrhs, a.data(), get_ld(a), b.data(), get_ld(b), work1, -1, info); 

        int lwork = int(real(work1[0]));
        array<T, 1> work(lwork);

        f77::gels(TRANS,get_n_rows(a), get_n_cols(a), nrhs, a.data(), get_ld(a), b.data(), get_ld(b), work.data(), lwork, info);

      } else {

        if constexpr (std::is_same_v<T, double>) {

          const char TRANS_T = ((TRANS=='N')?'T':'N');
        
          f77::gels(TRANS_T,get_n_rows(a), get_n_cols(a), nrhs, a.data(), get_ld(a), b.data(), get_ld(b), work1, -1, info);

          int lwork = int(work1[0]);
          array<T, 1> work(lwork);

          f77::gels(TRANS_T,get_n_rows(a), get_n_cols(a), nrhs, a.data(), get_ld(a), b.data(), get_ld(b), work.data(), lwork, info);

        } else if constexpr (std::is_same_v<T, dcomplex>) {

          const char TRANS_T = ((TRANS=='N')?'C':'N');

	  // conjugate a
	  for( auto& v: a ) v = std::conj(v);

          f77::gels(TRANS_T,get_n_rows(a), get_n_cols(a), nrhs, a.data(), get_ld(a), b.data(), get_ld(b), work1, -1, info);

          int lwork = int(real(work1[0]));
          array<T, 1> work(lwork);

          f77::gels(TRANS_T,get_n_rows(a), get_n_cols(a), nrhs, a.data(), get_ld(a), b.data(), get_ld(b), work.data(), lwork, info);

	  // conjugate a back
	  for( auto& v: a ) v = std::conj(v);

        } else
          static_assert(false and always_true<A>, "Internal logic error");

      }

      if (info) NDA_RUNTIME_ERROR << "Error in gels : info = " << info;
      return info;
    }
  }

} // namespace nda::lapack
