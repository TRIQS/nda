// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

#include "./bench_inputs.hpp"
#include "./benchmark_concepts.hpp"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace nda_bench {

  // Explicit value types for one registration; every operand uses the same type.
  template <typename... Ts>
  struct types {
    private:
    template <typename T>
    static constexpr std::size_t count = (std::size_t{0} + ... + std::size_t{std::same_as<T, Ts>});

    static constexpr bool supported =
       ((std::same_as<Ts, float> || std::same_as<Ts, double> || std::same_as<Ts, std::complex<float>> || std::same_as<Ts, std::complex<double>>)
        && ...);
    static constexpr bool unique = ((count<Ts> == 1) && ...);

    public:
    static_assert(sizeof...(Ts) > 0, "Select at least one benchmark value type");
    static_assert(supported, "Supported benchmark types: float, double, std::complex<float>, std::complex<double>");
    static_assert(unique, "Benchmark value types must not contain duplicates");

    using tuple_type = std::tuple<Ts...>;
  };

  inline constexpr std::int64_t min_dim = 16; // dimension along every axis
  // The sweep caps each owning operand, including unexposed slice elements.
  inline constexpr std::int64_t max_elements = std::int64_t{1} << 22;
  static_assert(min_dim > 0 && min_dim < max_elements);

  template <typename LHS, typename RHS>
  auto product_shape(LHS const &a, RHS const &b) {
    if constexpr (nda::Scalar<LHS>) {
      return b.shape();
    } else if constexpr (nda::Scalar<RHS>) {
      return a.shape();
    } else if constexpr (nda::get_algebra<LHS> == 'M' && nda::get_algebra<RHS> == 'M') {
      return std::array{a.extent(0), b.extent(1)};
    } else if constexpr (nda::get_algebra<LHS> == 'M' && nda::get_algebra<RHS> == 'V') {
      return std::array{a.extent(0)};
    } else {
      return a.shape();
    }
  }

  // Ops inherit these and override only what they do not support.
  struct op_defaults {
    static constexpr bool supports_complex = true;
    static constexpr bool supports_rank2   = true;
    static constexpr bool supports_rank3   = true;
    static constexpr bool supports_matrix  = true;

    template <typename First, typename... Rest>
    static auto result_shape(First const &first, Rest const &...rest) {
      if constexpr (nda::Array<First>) {
        return first.shape();
      } else {
        return result_shape(rest...);
      }
    }
  };

  namespace detail {

    // Build the operand tuple from storage, preserving references and slice views.
    template <typename ValueType, BenchmarkInput<ValueType>... Inputs>
    auto make_operands(std::tuple<typename Inputs::template storage_type<ValueType>...> &storage) {
      return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return std::tuple<decltype(Inputs::get(std::get<I>(storage)))...>{Inputs::get(std::get<I>(storage))...};
      }(std::index_sequence_for<Inputs...>{});
    }

    // Unpack the operand tuple and call the operation, preserving its return type.
    template <typename Operation, typename... OperandTypes>
    decltype(auto) invoke_op(std::tuple<OperandTypes...> const &operands) {
      static_assert(
         BenchmarkOperation<Operation, OperandTypes...>,
         "Benchmark operations must provide constexpr bool support flags, op returning an array or scalar, and result_shape for these operands");
      return std::apply([](auto const &...args) -> decltype(auto) { return Operation::op(args...); }, operands);
    }

    struct benchmark_stats {
      std::int64_t input_elements = 0;
      std::int64_t bytes          = 0;
    };

    template <typename OperandTuple>
    benchmark_stats measure_inputs(OperandTuple const &operands) {
      benchmark_stats stats;
      std::apply(
         [&](auto const &...x) {
           ((stats.input_elements = std::max(stats.input_elements, array_elements(x)), stats.bytes += logical_bytes(x)), ...);
         },
         operands);
      return stats;
    }

    template <typename ValueType, typename Operation, BenchmarkInput<ValueType>... Inputs>
    void run_benchmark(benchmark::State &state) {
      static_assert(
         BenchmarkOperation<Operation, decltype(Inputs::get(std::declval<typename Inputs::template storage_type<ValueType> &>()))...>,
         "Benchmark operations must provide constexpr bool support flags, op returning an array or scalar, and result_shape for these operands");
      static_assert(sizeof...(Inputs) > 0, "A benchmark needs at least one input descriptor");
      std::int64_t const N = state.range(0);
      std::mt19937_64 rng{42};

      std::tuple<typename Inputs::template storage_type<ValueType>...> storage{Inputs::template generate<ValueType>(N, rng)...};
      // References and views borrow storage; keep this tuple local and immutable.
      const auto operands = make_operands<ValueType, Inputs...>(storage);
      auto stats          = measure_inputs(operands);

      using ReturnType = decltype(invoke_op<Operation>(operands));
      using ResultType = std::remove_cvref_t<ReturnType>;

      // The result value type differs from ValueType for abs2 (always double), pow (promotes), isnan
      // (bool) and abs/real/imag on complex (real). Zero when the result is a scalar.
      if constexpr (nda::Array<ResultType> && (std::is_reference_v<ReturnType> || !nda::is_regular_v<ResultType>)) {
        // Shape discovery must not execute eager subexpressions such as matrix A * B + C.
        auto shape = std::apply([](auto const &...args) { return Operation::result_shape(args...); }, operands);
        nda::get_regular_t<ResultType> R(shape);
        R.as_array_view() = nda::get_value_t<ResultType>{}; // fault the pages in before the clock starts
        for (auto s : state) {
          // View assignment reuses storage, including for borrowed results.
          R() = invoke_op<Operation>(operands);
          benchmark::DoNotOptimize(R);
        }
        stats.bytes += logical_bytes(R);
      } else {
        ResultType R{};
        for (auto s : state) {
          R = invoke_op<Operation>(operands);
          benchmark::DoNotOptimize(R);
        }
        stats.bytes += logical_bytes(R);
      }

      state.SetItemsProcessed(state.iterations() * stats.input_elements);
      state.SetBytesProcessed(state.iterations() * stats.bytes);
      state.counters["NumberOfElements"] = stats.input_elements;
      state.counters["bytesize"]         = stats.bytes;
    }

    template <typename ValueType, BenchmarkInput<ValueType>... Inputs>
    void custom_range(benchmark::Benchmark *b) {
      for (std::int64_t N = min_dim; ((Inputs::template storage_size<ValueType>(N) <= max_elements) && ...); N *= 4) { b->Arg(N); }
    }

    template <typename ValueType, typename Operation, BenchmarkInput<ValueType> Input>
    constexpr bool supports_input() {
      using OperandType = std::remove_cvref_t<decltype(Input::get(std::declval<typename Input::template storage_type<ValueType> &>()))>;
      if constexpr (nda::is_complex_v<nda::get_value_t<OperandType>> && !Operation::supports_complex) {
        return false;
      } else if constexpr (nda::Array<OperandType>) {
        return (nda::get_rank<OperandType> != 2 || Operation::supports_rank2) && (nda::get_rank<OperandType> != 3 || Operation::supports_rank3)
           && (Input::algebra != 'M' || Operation::supports_matrix);
      } else {
        return true;
      }
    }

    // Name grammar <type>/<shape>/<op>, with the dimension appended by google-benchmark.
    template <typename ValueType, typename Operation, BenchmarkInput<ValueType>... Inputs>
    void register_case(std::string const &op_name) {
      static_assert(
         BenchmarkOperation<Operation, decltype(Inputs::get(std::declval<typename Inputs::template storage_type<ValueType> &>()))...>,
         "Benchmark operations must provide constexpr bool support flags, op returning an array or scalar, and result_shape for these operands");
      static_assert((supports_input<ValueType, Operation, Inputs>() && ...),
                    "Benchmark input type, rank, or algebra is unsupported by this operation");
      std::string shapes;
      ((shapes += (shapes.empty() ? "" : ",") + Inputs::template id<ValueType>()), ...);
      auto name = std::string(type_tag<ValueType>()) + "/" + shapes + "/" + op_name;
      benchmark::RegisterBenchmark(name, &run_benchmark<ValueType, Operation, Inputs...>)
         ->Apply(custom_range<ValueType, Inputs...>)
         ->ComputeStatistics("min", [](std::vector<double> const &v) { return *std::min_element(v.begin(), v.end()); })
         ->Unit(benchmark::kMicrosecond);
    }

    template <typename Operation, typename ValueTypes, typename... Inputs>
    struct input_registrar {
      using selected_types = typename ValueTypes::tuple_type;

      template <typename ValueType>
      static void register_type(std::string const &name) {
        constexpr bool supported = (supports_input<ValueType, Operation, Inputs>() && ...);
        static_assert(supported, "Selected benchmark type, rank, or algebra is unsupported by this operation");
        if constexpr (supported) { register_case<ValueType, Operation, Inputs...>(name); }
      }

      input_registrar(std::string const &name) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
          (register_type<std::tuple_element_t<I, selected_types>>(name), ...);
        }(std::make_index_sequence<std::tuple_size_v<selected_types>>{});
      }
    };

  } // namespace detail
} // namespace nda_bench

// Arguments next to ## are not expanded first. JOIN expands __COUNTER__ to a number,
// then JOIN_IMPL pastes it into the unique registration variable name.
#define NDA_BENCH_JOIN_IMPL(A, B) A##B
#define NDA_BENCH_JOIN(A, B) NDA_BENCH_JOIN_IMPL(A, B)
// Usage: NDA_BENCHMARK(op, "name", types<float, double>, input, ...).
// Keep the type list in __VA_ARGS__ so its template commas pass through intact.
#define NDA_BENCHMARK(OP_TYPE, NAME, ...)                                                                                                            \
  static ::nda_bench::detail::input_registrar<OP_TYPE, __VA_ARGS__> NDA_BENCH_JOIN(input_registrar_, __COUNTER__)(NAME);
