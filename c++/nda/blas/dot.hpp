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
#include "../concepts.hpp"
#include "../mem/address_space.hpp"
#include "tools.hpp"
#include "interface/cxx_interface.hpp"

namespace nda::blas {

  // --------
  template <typename X, typename Y>
  requires((Scalar<X> or MemoryVector<X>) and (Scalar<Y> or MemoryVector<X>))
  auto dot(X const &x, Y const &y) {
    if constexpr (Scalar<X> or Scalar<Y>) {
      return x * y;
    } else {
      static_assert(have_same_value_type_v<X, Y>, "Vectors must have same value type");
      static_assert(mem::have_same_addr_space_v<X, Y>, "Vectors must have same memory address space");
      static_assert(is_blas_lapack_v<get_value_t<X>>, "Vectors hold value_type incompatible with blas");

      EXPECTS(x.shape() == y.shape());

      if constexpr (mem::on_host<X>) {
        return f77::dot(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
      } else {
#if defined(NDA_HAVE_CUDA)
        return device::dot(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
#else
        static_assert(always_false<bool>," blas on device without gpu support! Compile for GPU. ");
        return std::decay_t<X>::value_type{0};
#endif
      }
    }
  }

  // --------
  template <typename X, typename Y>
  requires((Scalar<X> or MemoryVector<X>) and (Scalar<Y> or MemoryVector<X>))
  auto dotc(X const &x, Y const &y) {
    if constexpr (Scalar<X> or Scalar<Y>) {
      return conj(x) * y;
    } else {
      static_assert(have_same_value_type_v<X, Y>, "Vectors must have same value type");
      static_assert(mem::have_same_addr_space_v<X, Y>, "Vectors must have same memory address space");
      static_assert(is_blas_lapack_v<get_value_t<X>>, "Vectors hold value_type incompatible with blas");

      EXPECTS(x.shape() == y.shape());

      if constexpr (!is_complex_v<get_value_t<X>>) {
        return dot(x, y);
      } else if constexpr (mem::on_host<X>) {
        return f77::dotc(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
      } else {
#if defined(NDA_HAVE_CUDA)
        return device::dotc(x.size(), x.data(), x.indexmap().strides()[0], y.data(), y.indexmap().strides()[0]);
#else
        static_assert(always_false<bool>," blas on device without gpu support! Compile for GPU. ");
        return std::decay_t<X>::value_type{0};
#endif
      }
    }
  }

  // ------- Generic Impl -------

  template <bool star, typename X, typename Y>
  auto _dot_impl(X const &x, Y const &y) {
    EXPECTS(x.shape() == y.shape());
    long N = x.shape()[0];

    auto _conj = [](auto z) __attribute__((always_inline)) {
      if constexpr (star and is_complex_v<decltype(z)>) {
        return std::conj(z);
      } else
        return z;
    };

    if constexpr (has_layout_smallest_stride_is_one<X> and has_layout_smallest_stride_is_one<Y>) {
      if constexpr (is_regular_or_view_v<X> and is_regular_or_view_v<Y>) {
        auto *__restrict px = x.data();
        auto *__restrict py = y.data();
        auto res            = _conj(px[0]) * py[0];
        for (size_t i = 1; i < N; ++i) { res += _conj(px[i]) * py[i]; }
        return res;
      } else {
        auto res = _conj(x(_linear_index_t{0})) * y(_linear_index_t{0});
        for (long i = 1; i < N; ++i) { res += _conj(x(_linear_index_t{i})) * y(_linear_index_t{i}); }
        return res;
      }
    } else {
      auto res = _conj(x(0)) * y(0);
      for (long i = 1; i < N; ++i) { res += _conj(x(i)) * y(i); }
      return res;
    }
  }

  // --------
  template <typename X, typename Y>
  auto dot_generic(X const &x, Y const &y) {
    if constexpr (Scalar<X> or Scalar<Y>) {
      return x * y;
    } else {
      return _dot_impl<false>(x, y);
    }
  }

  // --------
  template <typename X, typename Y>
  auto dotc_generic(X const &x, Y const &y) {
    if constexpr (Scalar<X> or Scalar<Y>) {
      return conj(x) * y;
    } else {
      return _dot_impl<true>(x, y);
    }
  }

} // namespace nda::blas
