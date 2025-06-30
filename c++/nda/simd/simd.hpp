#pragma once

#include <xsimd/xsimd.hpp>

namespace nda {
  template<typename T>
  using native_simd = xsimd::batch<std::remove_cvref_t<T>>;

  template <typename T, size_t Width>
  using fixed_size_simd = xsimd::make_sized_batch_t<std::remove_cvref_t<T>, Width>;


} // namespace nda
