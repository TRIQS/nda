// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

#include <nda/nda.hpp>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nda::bench {
  // Explicit value types for one registration; every operand uses the same type.
  template <typename... Ts>
  struct types {
    template <typename T>
    static constexpr std::size_t count = (std::size_t{0} + ... + std::size_t{std::same_as<T, Ts>});

    static constexpr bool supported = (nda::is_any_of<Ts, float, double, std::complex<float>, std::complex<double>> and ...);
    static constexpr bool unique = ((count<Ts> == 1) and ...);

    static_assert(sizeof...(Ts) > 0, "Select at least one benchmark value type");
    static_assert(supported, "Supported benchmark types: float, double, std::complex<float>, std::complex<double>");
    static_assert(unique, "Benchmark value types must not contain duplicates");

    using tuple_type = std::tuple<Ts...>;
  };

  // Tag used in the benchmark name. The Python tools read it back from the case names: give a new
  // tag its C++ type in VALUE_TYPES in tools/benchmarks/common.py too, or the report's groupings
  // and the charts' legend show the bare tag.
  template <typename ValueType>
  consteval std::string_view type_tag() {
    if constexpr (std::same_as<ValueType, float>) {
      return "f32";
    } else if constexpr (std::same_as<ValueType, double>) {
      return "f64";
    } else if constexpr (std::same_as<ValueType, std::complex<float>>) {
      return "c64";
    } else if constexpr (std::same_as<ValueType, std::complex<double>>) {
      return "c128";
    } else {
      return "Unknown";
    }
  }

  // The default [1, 2) keeps inputs away from zero for division, log, and sqrt.
  template <std::int32_t Rank, char Algebra = 'A', typename Layout = nda::C_layout, auto Lower = 1.0, auto Upper = 2.0>
  struct array_input {
    static constexpr bool is_matrix     = Algebra == 'M' and Rank == 2;
    static constexpr bool is_array      = Algebra == 'A';
    static constexpr bool is_vector     = Algebra == 'V' and Rank == 1;

    static_assert(Rank > 0, "Input rank must be positive");
    static_assert(is_array or is_matrix or is_vector, "Input must be an array ('A'), a rank-2 matrix ('M'), or a rank-1 vector ('V')");

    static constexpr std::string_view name = is_matrix ? "M" : is_vector ? "V" : "A";

    static constexpr char algebra = Algebra;

    template <typename ValueType>
    using input_type = nda::basic_array<ValueType, Rank, Layout, Algebra, nda::heap<>>;

    static std::string id() {
      return std::string(name) + std::to_string(Rank) + "_" + std::string(Layout::to_string());
    }

    static std::int64_t input_size(std::int64_t N) {
      return static_cast<std::int64_t>(std::pow(N, Rank));
    }

    template <typename ValueType>
    static auto create_input(std::int64_t N) {
      using ArrayType = input_type<ValueType>;
      auto shape = nda::stdutil::make_initialized_array<Rank>(N);
      return ArrayType::rand(shape, Lower, Upper);
    }

    template <typename InputDataType>
    static auto &get_benchmark_argument(InputDataType &input_data) {
      return input_data;
    }
  };



  template <auto Lower = 1.0, auto Upper = 2.0>
  struct scalar_input {
    static constexpr std::string_view name = "S";

    static constexpr char algebra = 'S';

    template <typename ValueType>
    using input_type = ValueType;

    static std::string id() {
      return std::string(name);
    }

    static constexpr std::int64_t input_size(std::int64_t) {
      return 0;
    }

    template <typename ValueType>
    static ValueType create_input(std::int64_t) {
      using ArrayType = nda::stack_array<ValueType, 1>;
      return ArrayType::rand({1}, Lower, Upper)(0);
    }

    template <typename InputDataType>
    static auto &get_benchmark_argument(InputDataType &input_data) {
      return input_data;
    }
  };

  template <typename Input, std::int32_t Axis, std::int64_t Step, std::int64_t Start = 0>
  struct slice_input {
    static_assert(Axis >= 0, "Slice axis must be nonnegative");
    static_assert(Step > 0, "Slice step must be positive");
    static_assert(Start >= 0, "Slice start must be nonnegative");

    static constexpr std::string_view name = "slice";

    static constexpr char algebra = Input::algebra;

    template <typename ValueType>
    using input_type = typename Input::template input_type<ValueType>;

    static std::int64_t input_size(std::int64_t N) {
      return Input::input_size(N);
    }

    template <typename ValueType>
    static auto create_input(std::int64_t N) {
      return Input::template create_input<ValueType>(N);
    }

    static std::string id() {
      return Input::id() + "." + std::string(name) + "(axis=" + std::to_string(Axis) + ",start=" + std::to_string(Start)
         + ",step=" + std::to_string(Step) + ")";
    }

    template <typename InputDataType>
    static auto get_benchmark_argument(InputDataType &input_data) {
      auto &&a = Input::get_benchmark_argument(input_data);
      using OperandType = decltype(a);
      static constexpr int rank = nda::get_rank<OperandType>;

      static constexpr bool owns_memory    = nda::is_regular_v<std::remove_cvref_t<OperandType>>;
      static constexpr bool is_lvalue      = std::is_lvalue_reference_v<OperandType>;
      static constexpr bool lifetime_valid = not owns_memory or is_lvalue;
      static constexpr bool axis_in_bounds = Axis < rank;

      // A temporary owning array would be destroyed on return, leaving the slice dangling.
      static_assert(lifetime_valid, "Input::get_benchmark_argument must return owning arrays by lvalue reference when used with slice_input");
      static_assert(axis_in_bounds, "Slice axis exceeds input rank");
      auto const axis_extent = a.extent(Axis);
      if (Start >= axis_extent) { throw std::invalid_argument("Slice is empty at this benchmark size"); }

      auto range_for_dimension = [&]<std::size_t Dimension>() {
        if constexpr (Dimension == Axis) {
          return nda::range(Start, axis_extent, Step);
        } else {
          return nda::range::all;
        }
      };

      return [&]<std::size_t... Dimensions>(std::index_sequence<Dimensions...>) {
        return a(range_for_dimension.template operator()<Dimensions>()...);
      }(std::make_index_sequence<rank>{});
    }
  };

  template <typename Layout = nda::C_layout, auto Lower = 1.0, auto Upper = 2.0>
  using matrix_input = array_input<2, 'M', Layout, Lower, Upper>;

  template <typename Layout = nda::C_layout, auto Lower = 1.0, auto Upper = 2.0>
  using vector_input = array_input<1, 'V', Layout, Lower, Upper>;

  template <typename Input, std::int64_t Step = 2, std::int64_t Start = 0>
  using row_slice = slice_input<Input, 0, Step, Start>;

  template <typename Input, std::int64_t Step = 2, std::int64_t Start = 0>
  using col_slice = slice_input<Input, 1, Step, Start>;

  using types_real    = types<float, double>;
  using types_complex = types<std::complex<float>, std::complex<double>>;
  using types_single  = types<float, std::complex<float>>;
  using types_double  = types<double, std::complex<double>>;
  using types_all     = types<float, double, std::complex<float>, std::complex<double>>;

} // namespace nda::bench
