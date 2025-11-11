#pragma once
#include "./simd_fwd.hpp"
#include "../concepts.hpp"

#include <type_traits>

namespace nda {
  template <typename T>
  concept IsBasicArray = requires {
    []<typename ValueType, int Rank, typename Layout, char Algebra, typename ContainerPolicy>(
       basic_array<ValueType, Rank, Layout, Algebra, ContainerPolicy>) {}(std::declval<T>());
  };

  template <typename T>
  concept IsBasicArrayView = requires {
    []<typename ValueType, int Rank, typename Layout, char Algebra, typename AccessorPolicy, typename OwningPolicy>(
       basic_array_view<ValueType, Rank, Layout, Algebra, AccessorPolicy, OwningPolicy>) {}(std::declval<T>());
  };

  template <typename T>
  concept IsExprUnary = requires { []<char OP, Array A>(expr_unary<OP, A>) {}(std::declval<T>()); };

  template <typename T>
  concept IsExprCall = requires { []<typename F, Array... As>(expr_call<F, As...>) {}(std::declval<T>()); };

  template <typename T>
  concept IsExpr = requires { []<char OP, typename L, typename R>(expr<OP, L, R>) {}(std::declval<T>()); };

  namespace simd {
    /**
     * @brief Computes the compile-time "cost" of evaluating an expression template.
     *
     * This function recursively traverses the expression tree defined by the
     * template type `E` and calculates a cost based on the following rules:
     *
     * - **Cost 0:** Any non-array/expression type (e.g., scalars) has zero cost.
     * - **Cost 1:** Terminal nodes (basic_array or basic_array_view) have a cost of 1,
     * as they represent one leaf variable.
     * - **Cost 1 + children:** Unary, binary, and function-call expressions have a
     * cost of 1 (for the operation itself) plus the sum of the costs of all
     * their child expressions.
     *
     * This `consteval` function replaces the C++17 struct-based template
     * metaprogramming. It uses `if constexpr` to branch at compile time and
     * template lambdas to "pattern match" and extract sub-expression types.
     */
    template <typename A_in>
    static consteval size_t expr_cost() {
      using A = std::remove_cvref_t<A_in>;
      // Case 1 & 2: basic_array and basic_array_view (Terminals)
      if constexpr (IsBasicArray<A> or IsBasicArrayView<A>) {
        return 1;
      }
      // Case 3: expr_unary
      else if constexpr (IsExprUnary<A>) {
        return []<char OP, Array E>(std::type_identity<expr_unary<OP, E>>) { return 1 + expr_cost<E>(); }(std::type_identity<A>{});
      }
      // Case 4: expr_call
      else if constexpr (IsExprCall<A>) {
        return []<typename F, Array... As>(std::type_identity<expr_call<F, As...>>) { return 1 + (expr_cost<As>() + ...); }(std::type_identity<A>{});
      }
      // Case 5: expr (binary)
      else if constexpr (IsExpr<A>) {
        return []<char OP, typename L, typename R>(std::type_identity<expr<OP, L, R>>) {
          return 1 + expr_cost<L>() + expr_cost<R>();
        }(std::type_identity<A>{});
      }
      // Case 0: Base case for any other type (like int, double, etc.)
      else {
        return 0;
      }
    }

    /**
     * @brief Decides the evaluation strategy for an expression based on its cost.
     *
     * This class models the cost of evaluating an expression tree. It uses
     * `expression_cost<A>()` to calculate the actual cost of the tree.
     *
     * A `MAX_COST` threshold is defined. If the expression's cost is too high
     * (>= MAX_COST), the `emulate()` function will return true.
     *
     * This signals in the case where vectorized evaluation is not possible,
     * that a simple scalar evaluation is too costly according to expression cost model.
     * In such cases, we might opt for an "emulated" vectorized evaluation
     * for the entire expression.
     *
     */
    template <typename A>
    struct simd_cost_model {
      /*
       *TODO: This Threshold needs to be tuned in the future by running benchmarks on different expression trees with different Threshold costs
       * and see which threshold number gives the best performance. This threshold number is also dependent on how the cost of the expression tree
       * is calculated therefore when the model changes we have to rerun the benchmarks to decide the threshold cost.
       */

      static constexpr size_t MAX_COST = 16;
      static constexpr size_t cost() { return expr_cost<A>(); }
      static constexpr bool emulate() { return cost() >= MAX_COST; }
    };

    /**
     * @brief Checks at compile-time if an expression tree maintains a
     * consistent, known memory layout.
     *
     * This function recursively traverses the expression tree defined by `E`.
     *
     * - **false:** Default case. Any unknown type is assumed to not have a
     * consistent layout.
     * - **true:** Terminal nodes (basic_array, basic_array_view) always have a
     * known layout.
     * - **Recursive:** For unary expressions (expr_unary), the layout is
     * determined by the layout of its single child expression.
     * - **Direct Check:** For binary (expr) and function-call (expr_call)
     * expressions, the layout is determined by checking a separate trait,
     * `get_layout_info`, on the expression type itself.
     */
    template <typename A_in>
    static consteval bool has_same_layout() {
      using A = std::remove_cvref_t<A_in>;
      // Case 1 & 2: basic_array and basic_array_view (Terminals)
      if constexpr (IsBasicArray<A> or IsBasicArrayView<A>) {
        return true;
      }
      // Case 3: expr_unary
      else if constexpr (IsExprUnary<A>) {
        return []<char OP, Array E>(std::type_identity<expr_unary<OP, E>>) { return has_same_layout<E>(); }(std::type_identity<A>{});
      }
      // Case 4: expr_call
      else if constexpr (IsExprCall<A>) {
        return []<typename F, Array... As>(std::type_identity<expr_call<F, As...>>) {
          return get_layout_info<expr_call<F, As...>>.stride_order != static_cast<uint64_t>(-1);
        }(std::type_identity<A>{});
      }
      // Case 5: expr (binary)
      else if constexpr (IsExpr<A>) {
        return []<char OP, typename L, typename R>(std::type_identity<expr<OP, L, R>>) {
          return get_layout_info<expr<OP, L, R>>.stride_order != static_cast<uint64_t>(-1);
        }(std::type_identity<A>{});
      }
      // Case 0: Base case for any other type (like int, double, etc.)
      else {
        return false;
      }
    }
    /**
     * @brief A compile-time function to check if an expression tree is vectorizable.
     *
     * This function recursively traverses an expression tree (`A_in`) to determine if
     * it can be evaluated using SIMD operations. It returns true if and only if
     * every single node in the tree meets two criteria:
     *
     * 1. **Consistent Type:** All arrays, views, and scalar operands in the
     * expression must have the *exact same* scalar type (e.g., `double`).
     * This target type (`T`) is automatically deduced from the top-level
     * expression (`A_in`).
     *
     * 2. **Vectorizable Type:** The common scalar type must satisfy the
     * `Vectorizable` concept (e.g., it is an arithmetic type like `double`
     * or `int`, not `std::string`).
     *
     * @details
     * The logic is implemented as follows:
     * - **Base Cases (Arrays/Views):** Check if `ValueType` is `Vectorizable` and
     * `is_same_v<ValueType, T>`.
     * - **Recursive Cases (Unary/Call):** Check if all children recursively
     * satisfy `has_vectorizable_type<Child, T>()`.
     * - **Binary Case:** Checks if both children are valid. If one child is a
     * scalar, it ensures the scalar's type is also `is_same_v<ScalarType, T>`.
     * - **Default Case:** All other types are not vectorizable.
     */
    template <typename A_in, typename T_in = get_value_t<A_in>>
    static consteval bool has_vectorizable_type() {
      using A         = std::remove_cvref_t<A_in>;
      using T         = std::remove_cvref_t<T_in>;
      using ValueType = get_value_t<A>;
      // Case 1 & 2: basic_array and basic_array_view (Terminals)
      if constexpr (IsBasicArray<A> or IsBasicArrayView<A>) {
        return Vectorizable<ValueType> and std::is_same_v<ValueType, T>;
      }
      // Case 3: expr_unary
      else if constexpr (IsExprUnary<A>) {
        return []<char OP, Array E>(std::type_identity<expr_unary<OP, E>>) { return has_vectorizable_type<E>(); }(std::type_identity<A>{});
      }
      // Case 4: expr_call
      else if constexpr (IsExprCall<A>) {
        return []<typename F, Array... As>(std::type_identity<expr_call<F, As...>>) {
          return (has_vectorizable_type<As, T>() and ...);
        }(std::type_identity<A>{});
      }
      // Case 5: expr (binary)
      else if constexpr (IsExpr<A>) {
        return []<char OP, typename L, typename R>(std::type_identity<expr<OP, L, R>>) {
          if constexpr (is_scalar_v<L>) {
            return has_vectorizable_type<R, T>() and std::is_same_v<T, std::remove_cvref_t<L>>;
          } else if constexpr (is_scalar_v<R>) {
            return has_vectorizable_type<L, T>() and std::is_same_v<T, std::remove_cvref_t<R>>;
          } else {
            return has_vectorizable_type<L, T>() and has_vectorizable_type<R, T>();
          }
        }(std::type_identity<A>{});
      }
      // Case 0: Base case for any other type (like int, double, etc.)
      else {
        return false;
      }
    }

    /**
     * @brief Checks at compile-time if an expression tree can be evaluated using
     * a native SIMD load function.
     *
     * This function recursively traverses an expression tree (`A_in`) to determine if
     * it's eligible for a specialized  `load` function with type native_simd<T_in>.
     *
     * The logic is implemented as follows:
     * - **Target Type (T):** The scalar type (e.g., `double`) is deduced from the
     * top-level expression `A_in` and passed down recursively.
     * - **Base Cases (Arrays/Views):** The array/view must be vectorizable AND
     * its `ValueType` must match the target type `T`.
     * - **Unary Case:** Recurses on the single child, passing `T` along.
     * - **Call Case:** Checks two conditions:
     * 1. The function object `F` itself must be marked as natively loadable via
     * the `LoadWithNativeSimd` concept.
     * 2. All arguments `As...` must also recursively satisfy `has_load_function`.
     * - **Binary Case:**
     * - If one operand is scalar, it is ignored (its type does not matter),
     * and the check recurses only on the other (array) operand.
     * - If both are array expressions, both must recursively satisfy
     * `has_load_function`.
     * - **Default Case:** All other types are not loadable.
     */
    template <typename A_in, typename T_in = get_value_t<A_in>>
    static consteval bool has_load_function() {
      using A         = std::remove_cvref_t<A_in>;
      using T         = std::remove_cvref_t<T_in>;
      using ValueType = get_value_t<A>;
      // Case 1 & 2: basic_array and basic_array_view (Terminals)
      if constexpr (IsBasicArray<A> or IsBasicArrayView<A>) {
        return Vectorizable<ValueType> and std::is_same_v<ValueType, T>;
      }
      // Case 3: expr_unary
      else if constexpr (IsExprUnary<A>) {
        return []<char OP, Array E>(std::type_identity<expr_unary<OP, E>>) { return has_load_function<E, T>(); }(std::type_identity<A>{});
      }
      // Case 4: expr_call
      else if constexpr (IsExprCall<A>) {
        return []<typename F, Array... As>(std::type_identity<expr_call<F, As...>>) {
          return LoadWithNativeSimd<F, T, sizeof...(As)> and (has_load_function<As, T>() and ...);
        }(std::type_identity<A>{});
      }
      // Case 5: expr (binary)
      else if constexpr (IsExpr<A>) {
        return []<char OP, typename L, typename R>(std::type_identity<expr<OP, L, R>>) {
          if constexpr (is_scalar_v<L>) {
            return has_load_function<R, T>();
          } else if constexpr (is_scalar_v<R>) {
            return has_load_function<L, T>();
          } else {
            return has_load_function<L, T>() and has_load_function<R, T>();
          }
        }(std::type_identity<A>{});
      }
      // Case 0: Base case for any other type (like int, double, etc.)
      else {
        return false;
      }
    }

    struct vectorize_t {};
    struct emulate_t {};
    struct scalar_t {};

    inline static constexpr vectorize_t vectorize;
    inline static constexpr emulate_t emulate;
    inline static constexpr scalar_t scalar;

    template <typename A, typename T = get_value_t<A>>
    struct dispatch_policy {
      static constexpr bool same_layout        = has_same_layout<A>();
      static constexpr bool contiguous_layout  = has_contiguous_layout<A>;
      static constexpr bool vectorizable_types = has_vectorizable_type<A, T>();
      static constexpr bool load_available     = has_load_function<A, T>();

      static constexpr bool emulate = same_layout and contiguous_layout and vectorizable_types and !load_available and simd_cost_model<A>::emulate();
      static constexpr bool vectorize = same_layout and contiguous_layout and vectorizable_types and load_available;

      using type = std::conditional_t<vectorize, vectorize_t, std::conditional_t<emulate, emulate_t, scalar_t>>;
    };
    template <typename A, typename T = get_value_t<A>>
    using dispatch_policy_t = dispatch_policy<A, T>::type;
  } // namespace simd
} // namespace nda