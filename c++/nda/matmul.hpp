#pragma once
#include "blas/gemm.hpp"

namespace nda {

  /**
   * @tparam L NdArray with algebra 'M' 
   * @tparam R 
   * @param l : lhs
   * @param r : rhs
   * @return the matrix multiplication
   *   Implementation varies 
   */

  template <typename L, typename R>
  auto matmul(L &&l, R &&r) {
    using L_t = std::decay_t<L>; // L, R can be lvalue references
    using R_t = std::decay_t<R>;

    if (l.shape()[1] != r.shape()[0]) NDA_RUNTIME_ERROR << "Matrix product : dimension mismatch in matrix product " << l << " " << r;

    using promoted_type = decltype(get_value_t<L_t>{} * get_value_t<R_t>{});
    matrix<promoted_type> result(l.shape()[0], r.shape()[1]);

    // We need to make sure that l, and r are matrix or view before calling gemm, and that they have the right type
    // if not, we have to make a temporary
    static_assert(not blas::is_blas_lapack_v<long>, "");

    // if the type is double, dcomplex or a type understood by gemm
    if constexpr (blas::is_blas_lapack_v<promoted_type>) {

      auto as_container = [](auto const &a) -> decltype(auto) {
        using A = decltype(a);
        if constexpr (is_regular_or_view_v<A> and std::is_same_v<get_value_t<A>, promoted_type>)
          return a;
        else
          return matrix<promoted_type>{a};
      };

      // MSAN has no way to know that we are calling with beta = 0, hence
      // this is not necessaru
      // of course, in production code, we do NOT waste time to do this.
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      result = 0;
#endif
#endif

      blas::gemm(1, as_container(l), as_container(r), 0, result);
    } else {
      blas::gemm_generic(1, l, r, 0, result);
    }
    return result;
  }
} // namespace nda
