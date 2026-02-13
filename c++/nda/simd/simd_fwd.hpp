// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

#include "../concepts.hpp"

namespace nda {
  //Forward declaration
  template <typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>
  class basic_array;

  template <typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>
  class basic_array_view;

  template <char OP, Array A>
  struct expr_unary;

  template <char OP, ArrayOrScalar L, ArrayOrScalar R>
  struct expr;

  template <typename F, Array... A>
  struct expr_call;

} // namespace nda