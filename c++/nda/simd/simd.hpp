// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

#ifdef NDA_HAVE_XSIMD

#include <xsimd/xsimd.hpp>

#include <type_traits>

namespace nda {
  // TODO: create custom complex class.
  template <typename T>
  using native_simd = xsimd::batch<std::remove_cvref_t<T>>;

  template <typename T, size_t Width>
  using fixed_size_simd = xsimd::make_sized_batch_t<std::remove_cvref_t<T>, Width>;

} // namespace nda

#endif // NDA_HAVE_XSIMD