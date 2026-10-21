// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

#include "./bench_inputs.hpp"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace nda::bench {

  inline constexpr std::int64_t min_dim = 16; // dimension along every axis
  // The sweep caps each owning operand, including unexposed slice elements.
  inline constexpr std::int64_t max_elements = std::int64_t{1} << 22;
  static_assert(min_dim > 0 and min_dim < max_elements);

  template <nda::ArrayOrScalar LHS, nda::ArrayOrScalar RHS>
  auto product_shape(LHS const &a, RHS const &b) {
    if constexpr (nda::Scalar<LHS>) {
      return b.shape();
    } else if constexpr (nda::Scalar<RHS>) {
      return a.shape();
    } else if constexpr (nda::get_algebra<LHS> == 'M' and nda::get_algebra<RHS> == 'M') {
      return std::array{a.extent(0), b.extent(1)};
    } else if constexpr (nda::get_algebra<LHS> == 'M' and nda::get_algebra<RHS> == 'V') {
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

    // At least one of them has to be Array.
    template <nda::ArrayOrScalar First, nda::ArrayOrScalar... Rest>
    static auto result_shape(First const &first, Rest const &...rest) {
      if constexpr (nda::Array<First>) {
        return first.shape();
      } else {
        return result_shape(rest...);
      }
    }
  };

  // Runs one operation for one value type and an ordered list of input descriptors.
  //
  // Template parameters:
  //   ValueType: the element/scalar type shared by all inputs, e.g. double or std::complex<double>.
  //   Operation: a type providing static op(args...) and the supports_* flags, usually by inheriting
  //              op_defaults. result_shape(args...) supplies the output shape when storage is reused.
  //              An operation with a static target index instead provides op(output, args...).
  //   Inputs...: one descriptor per argument, in op's argument order, e.g. array_input<2>,
  //              scalar_input<>, matrix_input<>, or row_slice<matrix_input<>>.
  //
  // Registration and setup:
  //   register_case(name) checks the operation's supported types, ranks, and algebra, then registers
  //   a Google Benchmark case named <type>/<input ids>/<name>. custom_range appends dimensions
  //   N = 16, 64, 256, ... while every descriptor's input_size(N) stays within max_elements.
  //   input_size counts the full backing allocation, including elements hidden by a slice.
  //
  //   For each N, the constructor receives benchmark::State and reads N from state.range(0).
  //   Each Input::create_input<ValueType>(N) creates and initializes its owned data, stored in
  //   input_data as a tuple of Input::input_type<ValueType>. array_input<2> creates an N x N array
  //   with random values in [1, 2) by default; scalar_input<> creates one random scalar and ignores N.
  //   A slice descriptor creates its underlying input's full allocation at this stage.
  //
  //   Input::get_benchmark_argument(data) exposes the argument passed to Operation::op: a reference
  //   for a full array/scalar, or a view for a slice. make_benchmark_arguments() calls it for each tuple
  //   element and collects the results in benchmark_arguments. The owning input_data must outlive
  //   these references and views.
  //   Both input creation and argument extraction happen once, before the timed iteration loop.
  //
  // Execution (run selects a path at compile time):
  //   - Operation::target present: copy that argument into an owning output once, then repeatedly
  //     call op(output, args...). All original arguments are still passed, including the target.
  //   - Array expression/view/reference returned: allocate and initialize an owning output once
  //     using result_shape(args...), then evaluate output() = op(args...) each iteration.
  //   - Scalar or owning array returned: assign output = op(args...) each iteration; allocations
  //     performed inside op are therefore part of the measured work.
  //
  // Examples (op_add is the operation returning a + b):
  //   benchmark_runner<double, op_add, array_input<2>, scalar_input<>>
  //     owns an N x N double array and a double scalar, then times their addition into reused output.
  //   benchmark_runner<double, op_add, row_slice<matrix_input<>>, row_slice<matrix_input<>>>
  //     owns two N x N matrices and passes views of rows 0, 2, 4, ... to op_add.
  //   NDA_BENCHMARK(op_add, "add", types_double, array_input<2>, scalar_input<>)
  //     registers a runner for each value type in types_double (double and std::complex<double>).
  template <typename ValueType, typename Operation, typename... Inputs>
  struct benchmark_runner {
    static constexpr bool has_inputs = sizeof...(Inputs) > 0;
    static_assert(has_inputs, "A benchmark needs at least one input descriptor");

    // input_data_type owns the data; benchmark_arguments_type holds references or slice views passed to the computation.
    using input_data_type = std::tuple<typename Inputs::template input_type<ValueType>...>;
    using benchmark_arguments_type = std::tuple<decltype(Inputs::get_benchmark_argument(std::declval<typename Inputs::template input_type<ValueType> &>()))...>;

    benchmark::State &state; //NOLINT
    input_data_type input_data;
    // The run_* functions time a local copy of this tuple: read through *this, the arguments made GCC emit a slower
    // loop for some operations (f64/M2,S/add/16 took 143 ns instead of 84 ns).
    benchmark_arguments_type benchmark_arguments;

    explicit benchmark_runner(benchmark::State &benchmark_state)
       : state(benchmark_state), input_data(make_input_data(state.range(0))), benchmark_arguments(make_benchmark_arguments()) {}

    static input_data_type make_input_data(std::int64_t N) {
      nda::random::set_seed(42);
      return input_data_type{Inputs::template create_input<ValueType>(N)...};
    }

    benchmark_arguments_type make_benchmark_arguments() {
      return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return benchmark_arguments_type{Inputs::get_benchmark_argument(std::get<I>(input_data))...};
      }(std::index_sequence_for<Inputs...>{});
    }

    void run() {
      if constexpr (requires { Operation::target; }) { // Tensor operations
        run_in_place();
      } else {
        using ReturnType = decltype(std::apply([](auto const &...args) -> decltype(auto) { return Operation::op(args...); }, benchmark_arguments));
        using ResultType = std::remove_cvref_t<ReturnType>;

        static constexpr bool reuse_output = nda::Array<ResultType> and (std::is_reference_v<ReturnType> or not nda::is_regular_v<ResultType>);

        if constexpr (reuse_output) {
          run_with_reused_output<ResultType>();
        } else {
          run_with_returned_output<ResultType>();
        }
      }
    }

    void run_in_place() {
      // Copy the target once; repeated calls update the same output.
      using TargetType = std::remove_cvref_t<std::tuple_element_t<Operation::target, benchmark_arguments_type>>;
      auto benchmark_arguments_copy = benchmark_arguments;
      nda::get_regular_t<TargetType> output(std::get<Operation::target>(benchmark_arguments_copy));
      for (auto s : state) {
        std::apply([&output](auto const &...args) { Operation::op(output, args...); }, benchmark_arguments_copy);
        benchmark::DoNotOptimize(output);
      }
    }

    template <typename ResultType>
    void run_with_reused_output() {
      auto benchmark_arguments_copy = benchmark_arguments;
      // Discover the shape without executing eager subexpressions such as matrix A * B + C.
      auto shape = std::apply([](auto const &...args) { return Operation::result_shape(args...); }, benchmark_arguments_copy);
      nda::get_regular_t<ResultType> output(shape);
      output() = nda::get_value_t<ResultType>{}; // fault the pages in before the clock starts
      for (auto s : state) {
        output() = std::apply([](auto const &...args) -> decltype(auto) { return Operation::op(args...); }, benchmark_arguments_copy);
        benchmark::DoNotOptimize(output);
      }
    }

    template <typename ResultType>
    void run_with_returned_output() {
      // ResultType also covers scalar reductions and eager array results.
      auto benchmark_arguments_copy = benchmark_arguments;
      ResultType output{};
      for (auto s : state) {
        output = std::apply([](auto const &...args) -> decltype(auto) { return Operation::op(args...); }, benchmark_arguments_copy);
        benchmark::DoNotOptimize(output);
      }
    }

    static void custom_range(benchmark::Benchmark *b) {
      for (std::int64_t N = min_dim; ((Inputs::input_size(N) <= max_elements) and ...); N *= 4) { b->Arg(N); }
    }

    template <typename Input>
    static constexpr bool supports_input() {
      using InputDataType            = typename Input::template input_type<ValueType>;
      using OperandType              = std::remove_cvref_t<decltype(Input::get_benchmark_argument(std::declval<InputDataType &>()))>;
      constexpr bool value_supported = not nda::is_complex_v<ValueType> or Operation::supports_complex;

      if constexpr (nda::Array<OperandType>) {
        constexpr int rank               = nda::get_rank<OperandType>;
        constexpr bool rank2_supported   = rank != 2 or Operation::supports_rank2;
        constexpr bool rank3_supported   = rank != 3 or Operation::supports_rank3;
        constexpr bool algebra_supported = Input::algebra != 'M' or Operation::supports_matrix;
        return value_supported and rank2_supported and rank3_supported and algebra_supported;
      } else {
        return value_supported;
      }
    }

    // Name grammar <type>/<shape>/<op>, with the dimension appended by google-benchmark.
    static void register_case(std::string const &op_name) {
      constexpr bool inputs_supported = (supports_input<Inputs>() and ...);
      static_assert(inputs_supported, "Benchmark input type, rank, or algebra is unsupported by this operation");
      if constexpr (inputs_supported) {
        std::string shapes;
        ((shapes += (shapes.empty() ? "" : ",") + Inputs::id()), ...);
        auto name = std::string(type_tag<ValueType>()) + "/" + shapes + "/" + op_name;
        benchmark::RegisterBenchmark(name,
                                     [](benchmark::State &state) {
                                       benchmark_runner runner{state};
                                       runner.run();
                                     })
           ->Apply(&benchmark_runner::custom_range)
           ->ComputeStatistics("min", [](std::vector<double> const &v) { return *std::min_element(v.begin(), v.end()); })
           ->Unit(benchmark::kMicrosecond);
      }
    }
  };

  namespace detail {

    template <typename Operation, typename ValueTypes, typename... Inputs>
    struct input_registrar {
      using value_types_tuple = typename ValueTypes::tuple_type;

      input_registrar(std::string const &name) {
        [&]<std::size_t... I>(std::index_sequence<I...>) {
          (benchmark_runner<std::tuple_element_t<I, value_types_tuple>, Operation, Inputs...>::register_case(name), ...);
        }(std::make_index_sequence<std::tuple_size_v<value_types_tuple>>{});
      }
    };

  } // namespace detail
} // namespace nda::bench

// Arguments next to ## are not expanded first. JOIN expands __COUNTER__ to a number,
// then JOIN_IMPL pastes it into the unique registration variable name.
#define NDA_BENCH_JOIN_IMPL(A, B) A##B
#define NDA_BENCH_JOIN(A, B) NDA_BENCH_JOIN_IMPL(A, B)
// Keep the type list in __VA_ARGS__ so its template commas pass through intact.
#define NDA_BENCHMARK(OP_TYPE, NAME, ...)                                                                                                            \
  static ::nda::bench::detail::input_registrar<OP_TYPE, __VA_ARGS__> NDA_BENCH_JOIN(input_registrar_, __COUNTER__)(NAME);
