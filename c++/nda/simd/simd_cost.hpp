#pragma once
#include "../concepts.hpp"

#include <type_traits>

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
  namespace simd {
    //Forward declaration
    template <typename A>
    struct expr_cost;

    namespace detail {

      template <typename A>
      struct expr_cost_impl {
        static constexpr size_t value = 0;
      };

      template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename ContainerPolicy>
      struct expr_cost_impl<basic_array<ValueType, Rank, LayoutPolicy, Algebra, ContainerPolicy>> {
        static constexpr size_t value = 1;
      };

      template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename AccessorPolicy, typename OwningPolicy>
      struct expr_cost_impl<basic_array_view<ValueType, Rank, LayoutPolicy, Algebra, AccessorPolicy, OwningPolicy>> {
        static constexpr size_t value = 1;
      };

      template <char OP, Array A>
      struct expr_cost_impl<expr_unary<OP, A>> {
        static constexpr size_t value = 1 + expr_cost<A>::value;
      };

      template <typename F, Array... As>
      struct expr_cost_impl<expr_call<F, As...>> {
        static constexpr size_t value = 1 + (expr_cost<As>::value + ...);
      };

      template <char OP, typename L, typename R>
      struct expr_cost_impl<expr<OP, L, R>> {
        static constexpr size_t value = 1 + expr_cost<L>::value + expr_cost<R>::value;
      };

    } // namespace detail

    template <typename A>
    struct expr_cost {
      static constexpr size_t value = detail::expr_cost_impl<std::remove_cvref_t<A>>::value;
    };
    template <typename A>
    inline constexpr size_t expr_cost_v = expr_cost<A>::value;

    template <typename A>
    struct simd_cost_model {
      static constexpr size_t MAX_COST = 16; // TODO: tune it
      static constexpr size_t cost() { return expr_cost<A>::value; }
      static constexpr bool emulate() { return cost() >= MAX_COST; }
    };


  } // namespace simd
} // namespace nda