// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once
#include "../concepts.hpp"

#ifdef NDA_HAVE_XSIMD

namespace nda::simd {
  template <typename Derived, Vectorizable T>
  struct mock_simd {
    using value_t = T;
    using simd_t  = native_simd<T>;

    template <typename... Args>
    FORCEINLINE value_t operator()(Args &&...args) const {
      static_assert((std::is_same_v<value_t, std::remove_cvref_t<Args>> and ...), "All types have to be the same.");
      return static_cast<const Derived *>(this)->operator()(std::forward<Args>(args)...);
    }

    private:
    FORCEINLINE std::array<value_t, simd_t::size> convert_simd_to_array(const simd_t &a) const {
      alignas(simd_t::arch_type::alignment()) std::array<T, simd_t::size> result;
      a.store_aligned(result.data());
      return result;
    }

    template <size_t... Is, typename... Args>
    FORCEINLINE auto make_array_tuple(std::index_sequence<Is...>, const std::tuple<Args...> &args_tuple) const {
      return std::make_tuple(convert_simd_to_array(std::get<Is>(args_tuple))...);
    }

    template <size_t... Is, typename... Args>
    FORCEINLINE auto apply_function(std::index_sequence<Is...>, const std::tuple<Args...> &array_tuple) const {
      auto build = [&]<size_t... Js>(std::index_sequence<Js...>) {
        auto compute_one_element = [&](size_t i) -> T { return static_cast<const Derived *>(this)->operator()(std::get<Is>(array_tuple)[i]...); };

        alignas(simd_t::arch_type::alignment()) std::array<T, simd_t::size> result_array = {compute_one_element(Js)...};

        return result_array;
      };

      return build(std::make_index_sequence<simd_t::size>{});
    }

    public:
    template <typename... Args>
    FORCEINLINE simd_t load(Args &&...args) const {
      static_assert((std::is_same_v<simd_t, std::remove_cvref_t<Args>> and ...), "All types have to be the same.");
      constexpr size_t args_size        = sizeof...(Args);
      std::tuple<Args &&...> args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
      auto array_tuple                  = make_array_tuple(std::make_index_sequence<args_size>{}, args_tuple);
      alignas(simd_t::arch_type::alignment()) std::array<value_t, simd_t::size> result_array;
      result_array = apply_function(std::make_index_sequence<args_size>{}, array_tuple);
      return simd_t::load_aligned(result_array.data());
    }
  };
} // namespace nda::simd

#endif // NDA_HAVE_XSIMD