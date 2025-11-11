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