#pragma once

#include "tools.hpp"
#include "blas_interface/cxx_interface.hpp"

namespace nda::blas {

  template <typename X, typename Y>
  auto dot(X const &x, Y const &y) {
    EXPECTS(x.size() == y.size());
    long N = x.size();

    if constexpr (has_layout_smallest_stride_is_one<X> and has_layout_smallest_stride_is_one<Y>) {
      auto *__restrict px = x.data_start();
      auto *__restrict py = y.data_start();
      auto res            = px[0] * py[0];
      for (size_t i = 1; i < N; ++i) { res += px[i] * py[i]; }
      return res;
    } else {
      auto res = x(0) * y(0);
      for (size_t i = 1; i < N; ++i) { res += x(i) * y(i); }
      return res;
    }
  }

  template <typename X, typename Y>
  auto dotc(X const &x, Y const &y) {
    EXPECTS(x.size() == y.size());
    long N = x.size();

    if constexpr (has_layout_smallest_stride_is_one<X> and has_layout_smallest_stride_is_one<Y>) {
      auto *__restrict px = x.data_start();
      auto *__restrict py = y.data_start();
      auto res            = std::conj(px[0]) * py[0];
      for (size_t i = 1; i < N; ++i) { res += std::conj(px[i]) * py[i]; }
      return res;
    } else {
      auto res = x(0) * y(0);
      for (size_t i = 1; i < N; ++i) { res += std::conj(x(i)) * y(i); }
      return res;
    }
  }
} // namespace nda::blas
