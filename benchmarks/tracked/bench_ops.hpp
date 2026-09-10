// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

// Self-contained rather than including ../bench_common.hpp: that header has no include
// guard and defines the namespace-scope globals `_` and `___`, which nothing here uses.
#include <nda/nda.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

using namespace nda;

namespace nda_bench {

  inline constexpr std::int64_t min_dim      = 4; // dimension along every axis
  inline constexpr std::int64_t max_elements = std::int64_t{1} << 24;
  static_assert(min_dim > 0 and min_dim < max_elements);

  // Tag used in the benchmark name.
  template <typename T>
  struct type_tag;

  template <>
  struct type_tag<float> {
    static constexpr std::string_view name = "f32";
  };
  template <>
  struct type_tag<double> {
    static constexpr std::string_view name = "f64";
  };
  template <>
  struct type_tag<std::complex<float>> {
    static constexpr std::string_view name = "c64";
  };
  template <>
  struct type_tag<std::complex<double>> {
    static constexpr std::string_view name = "c128";
  };

  // rand() fills every component with [0,1); each op maps that into the range its own
  // expression needs. The default keeps values away from zero so division, log and sqrt
  // behave, but that is wrong for asin/acos (domain [-1,1]) and product (overflows), which
  // override it.
  struct input_range {
    double scale  = 1.0;
    double offset = 1.0; // default: [1, 2)
  };

  // Ops inherit these and override only what they do not support.
  struct op_defaults {
    static constexpr bool supports_complex = true;
    static constexpr bool supports_rank3   = true;
    static constexpr bool supports_matrix  = true;
    static constexpr input_range inputs    = {};
  };

  namespace detail {

    // Number of parameters of a function type.
    template <typename F>
    struct parameter_count;
    template <typename R, typename... Args>
    struct parameter_count<R(Args...)> : std::integral_constant<std::int32_t, sizeof...(Args)> {};

    // How many operands Op::op consumes.
    template <typename Op, typename ArrayT>
    inline constexpr std::int32_t operand_count = parameter_count<decltype(Op::template op<ArrayT>)>::value;

    template <typename Op, typename Pool, std::size_t... I>
    decltype(auto) invoke_op(Pool &pool, std::index_sequence<I...>) {
      return Op::op(pool[I]...);
    }

    constexpr std::int64_t ipow(std::int64_t base, std::int32_t exp) {
      std::int64_t p = 1;
      for (std::int32_t i = 0; i < exp; ++i) { p *= base; }
      return p;
    }

    template <std::int32_t Rank>
    void custom_range(benchmark::Benchmark *b) {
      static_assert(ipow(min_dim, Rank) <= max_elements, "nda_bench: min_dim^Rank exceeds max_elements; the sweep would be empty");
      for (std::int64_t n = min_dim; ipow(n, Rank) <= max_elements; n *= 2) { b->Args({n}); }
    }

    template <typename T, typename Op, typename ArrayT>
    void run_benchmark(benchmark::State &state) {

      constexpr std::int32_t n_ops = operand_count<Op, ArrayT>;

      std::int64_t const N = state.range(0); // dimension: the extent along every axis

      std::array<std::int64_t, get_rank<ArrayT>> shape;
      shape.fill(N);

      std::array<ArrayT, n_ops> pool;
      for (auto &x : pool) { x = ArrayT::rand(shape); }

      constexpr T in_scale = [] {
        using Re = remove_complex_t<T>;
        if constexpr (is_complex_v<T>) {
          return T{static_cast<Re>(Op::inputs.scale), Re{0}};
        } else {
          return static_cast<T>(Op::inputs.scale);
        }
      }();
      constexpr T in_offset = [] {
        using Re = remove_complex_t<T>;
        if constexpr (is_complex_v<T>) {
          auto const o = static_cast<Re>(Op::inputs.offset);
          return T{o, o};
        } else {
          return static_cast<T>(Op::inputs.offset);
        }
      }();

      for (auto &x : pool) {
        auto v = x.as_array_view();
        v = v * in_scale + in_offset;
      }

      constexpr auto seq = std::make_index_sequence<n_ops>{};

      using op_result_t = decltype(invoke_op<Op>(pool, seq));
      using result_t    = std::remove_cvref_t<op_result_t>;

      // The result value type differs from T for abs2 (always double), pow (promotes), isnan
      // (bool) and abs/real/imag on complex (real). Zero when the result is a scalar.
      constexpr std::int64_t result_bytes = [] {
        if constexpr (Array<result_t>) {
          return static_cast<std::int64_t>(sizeof(get_value_t<result_t>));
        } else {
          return std::int64_t{0};
        }
      }();

      if constexpr (Array<result_t> and (not MemoryArray<result_t> or std::is_reference_v<op_result_t>)) {
        get_regular_t<result_t> R(shape);
        R = get_value_t<result_t>{}; // fault the pages in before the clock starts
        for (auto s : state) {
          // View assignment reuses storage, including for borrowed results.
          R() = invoke_op<Op>(pool, seq);
          benchmark::DoNotOptimize(R);
        }
      } else {
        result_t R{};
        for (auto s : state) {
          R = invoke_op<Op>(pool, seq);
          benchmark::DoNotOptimize(R);
        }
      }

      std::int64_t const n_elements = pool[0].size();
      std::int64_t const bytes      = n_elements * (n_ops * static_cast<std::int64_t>(sizeof(T)) + result_bytes);
      state.SetItemsProcessed(state.iterations() * n_elements);
      state.SetBytesProcessed(state.iterations() * bytes);
      state.counters["NumberOfElements"] = double(n_elements);
      state.counters["bytesize"]         = double(bytes);
    }

    template <typename Op>
    struct registrar {

      // Name grammar <type>/<shape>/<op>, with the dimension appended by google-benchmark.
      template <typename T, typename ArrayT>
      void reg(std::string const &op_name, char const *shape) {
        BENCHMARK_TEMPLATE(run_benchmark, T, Op, ArrayT)
           ->Name(std::string(type_tag<T>::name) + "/" + shape + "/" + op_name)
           ->Apply(custom_range<get_rank<ArrayT>>)
           ->Unit(benchmark::kMicrosecond);
      }

      template <typename T>
      void reg_shapes(std::string const &op_name) {
        reg<T, array<T, 2>>(op_name, "a2");
        if constexpr (Op::supports_rank3) { reg<T, array<T, 3>>(op_name, "a3"); }
        if constexpr (Op::supports_matrix) { reg<T, matrix<T>>(op_name, "m2"); }
      }

      registrar(std::string const &op_name) {
        reg_shapes<float>(op_name);
        reg_shapes<double>(op_name);
        if constexpr (Op::supports_complex) {
          reg_shapes<std::complex<float>>(op_name);
          reg_shapes<std::complex<double>>(op_name);
        }
      }
    };

  } // namespace detail

} // namespace nda_bench

// Register one op across every supported value type and shape.
#define NDA_BENCHMARK_ALL_TYPES(OP_TYPE, NAME) static ::nda_bench::detail::registrar<OP_TYPE> registrar_##OP_TYPE(NAME);
