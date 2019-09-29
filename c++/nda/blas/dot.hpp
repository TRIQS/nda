#pragma once
#include "tools.hpp"
#include "interface/cxx_interface.hpp"

namespace nda::blas {

  // reimplement rather use blas.
  // there was a major issue with dotc on OS X. in triqs::arrays. Keep it this way.
  template <bool star, typename X, typename Y>
  auto _dot_impl(X const &x, Y const &y) {
    EXPECTS(x.shape() == y.shape());
    long N = x.extent(0);

    auto _conj = [](auto z) __attribute__((always_inline)) {
      if constexpr (star and is_complex_v<decltype(z)>) {
        return std::conj(z);
      } else
        return z;
    };

    if constexpr (has_layout_smallest_stride_is_one<X> and has_layout_smallest_stride_is_one<Y>) {
      if constexpr (is_regular_or_view_v<X> and is_regular_or_view_v<Y>) {
        auto *__restrict px = x.data_start();
        auto *__restrict py = y.data_start();
        auto res            = _conj(px[0]) * py[0];
        for (size_t i = 1; i < N; ++i) { res += _conj(px[i]) * py[i]; }
        return res;
      } else {
        auto res = _conj(x(_linear_index_t{0})) * y(_linear_index_t{0});
        for (long i = 1; i < N; ++i) { res += _conj(x(_linear_index_t{i})) * y(_linear_index_t{i}); }
        return res;
      }
    } else {
      auto res = x(0) * y(0);
      for (long i = 1; i < N; ++i) { res += _conj(x(i)) * y(i); }
      return res;
    }
  }

  // --------
  template <typename X, typename Y>
  auto dot(X const &x, Y const &y) {
    EXPECTS(x.shape() == y.shape());
    return _dot_impl<false>(x, y);
  }

  // --------
  template <typename X, typename Y>
  auto dotc(X const &x, Y const &y) {
    EXPECTS(x.shape() == y.shape());
    return _dot_impl<true>(x, y);
  }

} // namespace nda::blas
