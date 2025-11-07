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
      static constexpr size_t cost() { return expr_cost_v<A>; }
      static constexpr bool emulate() { return cost() >= MAX_COST; }
    };

    namespace detail {
      template <typename A>
      struct has_same_layout {
        static constexpr bool value = false;
      };

      template <typename A>
        requires(!std::is_same_v<A, std::remove_cvref_t<A>>)
      struct has_same_layout<A> {
        static constexpr bool value = has_same_layout<std::remove_cvref_t<A>>::value;
      };

      template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename ContainerPolicy>
      struct has_same_layout<basic_array<ValueType, Rank, LayoutPolicy, Algebra, ContainerPolicy>> {
        static constexpr bool value = true;
      };

      template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename AccessorPolicy, typename OwningPolicy>
      struct has_same_layout<basic_array_view<ValueType, Rank, LayoutPolicy, Algebra, AccessorPolicy, OwningPolicy>> {
        static constexpr bool value = true;
      };

      template <typename F, Array... As>
      struct has_same_layout<expr_call<F, As...>> {
        static constexpr bool value = get_layout_info<expr_call<F, As...>>.stride_order != static_cast<uint64_t>(-1);
      };

      template <char OP, Array A>
      struct has_same_layout<expr_unary<OP, A>> {
        static constexpr bool value = has_same_layout<A>::value;
      };

      template <char OP, typename L, typename R>
      struct has_same_layout<expr<OP, L, R>> {
        static constexpr bool value = get_layout_info<expr<OP, L, R>>.stride_order != static_cast<uint64_t>(-1);
      };

      template <typename A>
      inline constexpr bool has_same_layout_v = has_same_layout<A>::value;

      template <typename A, typename T = get_value_t<A>>
      struct has_vectorizable_type {
        static constexpr bool value = false;
      };

      template <typename A, typename T>
        requires(!std::is_same_v<A, std::remove_cvref_t<A>>)
      struct has_vectorizable_type<A, T> {
        static constexpr bool value = has_vectorizable_type<std::remove_cvref_t<A>, T>::value;
      };

      template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename ContainerPolicy, typename T>
      struct has_vectorizable_type<basic_array<ValueType, Rank, LayoutPolicy, Algebra, ContainerPolicy>, T> {
        static constexpr bool value = Vectorizable<ValueType> and std::is_same_v<ValueType, T>;
      };

      template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename AccessorPolicy, typename OwningPolicy, typename T>
      struct has_vectorizable_type<basic_array_view<ValueType, Rank, LayoutPolicy, Algebra, AccessorPolicy, OwningPolicy>, T> {
        static constexpr bool value = Vectorizable<ValueType> and std::is_same_v<ValueType, T>;
      };

      template <typename F, Array... As, typename T>
      struct has_vectorizable_type<expr_call<F, As...>, T> {
        static constexpr bool value = (has_vectorizable_type<As, T>::value and ...);
      };

      template <char OP, Array A, typename T>
      struct has_vectorizable_type<expr_unary<OP, A>, T> {
        static constexpr bool value = has_vectorizable_type<A, T>::value;
      };

      template <char OP, typename L, typename R, typename T>
      struct has_vectorizable_type<expr<OP, L, R>, T> {
        static constexpr bool value = (is_scalar_v<L>    ? (has_vectorizable_type<R, T>::value and std::is_same_v<T, std::remove_cvref_t<L>>) :
                                          is_scalar_v<R> ? (has_vectorizable_type<L, T>::value and std::is_same_v<T, std::remove_cvref_t<R>>) :
                                                           (has_vectorizable_type<L, T>::value and has_vectorizable_type<R, T>::value));
      };
      template <typename A, typename T = get_value_t<A>>
      inline constexpr bool has_vectorizable_type_v = has_vectorizable_type<A, T>::value;

      template <typename A, typename T = get_value_t<A>>
      struct has_load_function {
        static constexpr bool value = false;
      };

      template <typename A, typename T>
        requires(!std::is_same_v<A, std::remove_cvref_t<A>>)
      struct has_load_function<A, T> {
        static constexpr bool value = has_load_function<std::remove_cvref_t<A>, T>::value;
      };

      template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename ContainerPolicy, typename T>
      struct has_load_function<basic_array<ValueType, Rank, LayoutPolicy, Algebra, ContainerPolicy>, T> {
        static constexpr bool value = Vectorizable<ValueType> and std::is_same_v<ValueType, T>;
      };

      template <typename ValueType, int Rank, typename LayoutPolicy, char Algebra, typename AccessorPolicy, typename OwningPolicy, typename T>
      struct has_load_function<basic_array_view<ValueType, Rank, LayoutPolicy, Algebra, AccessorPolicy, OwningPolicy>, T> {
        static constexpr bool value = Vectorizable<ValueType> and std::is_same_v<ValueType, T>;
      };

      template <typename F, Array... As, typename T>
      struct has_load_function<expr_call<F, As...>, T> {
        static constexpr bool value = LoadWithNativeSimd<F, T, sizeof...(As)> and (has_load_function<As, T>::value and ...);
      };

      template <char OP, Array A, typename T>
      struct has_load_function<expr_unary<OP, A>, T> {
        static constexpr bool value = has_load_function<A, T>::value;
      };

      template <char OP, typename L, typename R, typename T>
      struct has_load_function<expr<OP, L, R>, T> {
        static constexpr bool value = (is_scalar_v<L>    ? (has_load_function<R, T>::value) :
                                          is_scalar_v<R> ? (has_load_function<L, T>::value) :
                                                           (has_load_function<L, T>::value and has_load_function<R, T>::value));
      };
      template <typename A, typename T = get_value_t<A>>
      inline constexpr bool has_load_function_v = has_load_function<A, T>::value;

    } // namespace detail

    struct vectorize_t {};
    struct emulate_t {};
    struct scalar_t {};

    inline static constexpr vectorize_t vectorize;
    inline static constexpr emulate_t emulate;
    inline static constexpr scalar_t scalar;

    template <typename A, typename T = get_value_t<A>>
    struct dispatch_policy {
      static constexpr bool same_layout         = detail::has_same_layout_v<A>;
      static constexpr bool contiguous_layout_v = has_contiguous_layout<A>;
      static constexpr bool vectorizable_types  = detail::has_vectorizable_type_v<A, T>;
      static constexpr bool load_available      = detail::has_load_function_v<A, T>;

      static constexpr bool emulate =
         same_layout and contiguous_layout_v and vectorizable_types and !load_available and simd_cost_model<A>::emulate();
      static constexpr bool vectorize = same_layout and contiguous_layout_v and vectorizable_types and load_available;

      using type = std::conditional_t<vectorize, vectorize_t, std::conditional_t<emulate, emulate_t, scalar_t>>;
    };
    template <typename A, typename T = get_value_t<A>>
    using dispatch_policy_t = dispatch_policy<A, T>::type;
  } // namespace simd
} // namespace nda