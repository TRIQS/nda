#pragma once
#include "./nda_complex.hpp"
#include <xsimd/xsimd.hpp>

#include <type_traits>

namespace nda {
  // TODO: create custom complex class.
  template <typename T>
  struct is_complex_impl : std::false_type {};
  template <typename T>
  struct is_complex_impl<std::complex<T>> : std::true_type {};

  namespace detail {
    template <typename T>
    struct native_simd_impl {
      using type = xsimd::batch<T>;
    };

    template <typename T>
      requires(std::is_same_v<std::remove_cvref_t<T>, std::complex<float>> or std::is_same_v<std::remove_cvref_t<T>, std::complex<double>>)
    struct native_simd_impl<T> {
      using type = nda::complex_batch<T>;
    };
  } // namespace detail

  template <typename T>
  using native_simd = typename detail::native_simd_impl<std::remove_cvref_t<T>>::type;
  template <typename T, size_t Width>
  using fixed_size_simd = xsimd::make_sized_batch_t<std::remove_cvref_t<T>, Width>;

} // namespace nda