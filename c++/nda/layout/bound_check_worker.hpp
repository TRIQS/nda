// Copyright (c) 2018--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a way to check the bounds when accessing elements/slices of an array or a view.
 */

#pragma once

#include "./range.hpp"

#include <cstdint>
#include <stdexcept>
#include <sstream>

namespace nda::detail {

  // Check the bounds when accessing single elements or slices of an array/view.
  struct bound_check_worker {
    // Shape of the array/view.
    long const *lengths{};

    // Error code to store the positions of the arguments which are out of bounds.
    uint32_t error_code = 0;

    // Number of dimensions that are covered by a given nda::ellipsis.
    int ellipsis_loss = 0;

    // Current dimension to be checked.
    int N = 0;

    // Check if the given index is within the bounds of the array/view.
    void check_current_dim(long idx) {
      if ((idx < 0) or (idx >= lengths[N])) { error_code += 1ul << N; }
      ++N;
    }

    // Check if the given nda::range is within the bounds of the array/view.
    void check_current_dim(range const &r) {
      if (r.size() > 0) {
        auto first_idx = r.first();
        auto last_idx  = first_idx + (r.size() - 1) * r.step();
        if (first_idx < 0 or first_idx >= lengths[N] or last_idx < 0 or last_idx >= lengths[N]) error_code += 1ul << N;
      }
      ++N;
    }

    // Check the bounds when an nda::range::all_t is encountered (no need to check anything).
    void check_current_dim(range::all_t) { ++N; }

    // Check the bounds when an nda::ellipsis is encountered (no need to check anything).
    void check_current_dim(ellipsis) { N += ellipsis_loss + 1; }

    // Accumulate an error message for the current dimension and index.
    void accumulate_error_msg(std::stringstream &fs, long idx) {
      if (error_code & (1ull << N)) fs << "Argument " << N << " = " << idx << " is not within [0," << lengths[N] << "[.\n";
      N++;
    }

    // Accumulate an error message for the current dimension and nda::range.
    void accumulate_error_msg(std::stringstream &fs, range const &r) {
      if (error_code & (1ull << N)) fs << "Argument " << N << " = " << r << " is not within [0," << lengths[N] << "[.\n";
      ++N;
    }

    // Accumulate an error message for the current dimension and nda::range::all_t.
    void accumulate_error_msg(std::stringstream &, range::all_t) { ++N; }

    // Accumulate an error message for the current dimension and nda::ellipsis.
    void accumulate_error_msg(std::stringstream &, ellipsis) { N += ellipsis_loss + 1; }
  };

} // namespace nda::detail

namespace nda {

  /**
   * @ingroup layout_utils
   * @brief Check if the given indices/arguments are within the bounds of an array/view.
   *
   * @details It uses the shape and rank of the array/view to check if the given arguments are within bounds. If an
   * error is detected, it throws a std::runtime_error with the error message.
   *
   * @tparam Args Types of the arguments to be checked.
   * @param rank Rank of the array/view.
   * @param lengths Shape of the array/view.
   * @param args Arguments (`long` indices, `nda::range`, `nda::range::all_t` or nda::ellipsis) to be checked.
   */
  template <typename... Args>
  void assert_in_bounds(int rank, long const *lengths, Args const &...args) {
    // initialize the bounds checker
    detail::bound_check_worker w{lengths};

    // number of dimensions that are covered by an nda::ellipsis
    w.ellipsis_loss = rank - sizeof...(Args);

    // check the bounds on each argument/index
    (w.check_current_dim(args), ...);

    // if no error, stop here
    if (!w.error_code) return;

    // accumulate error message and throw
    w.N = 0;
    std::stringstream fs;
    (w.accumulate_error_msg(fs, args), ...);
    throw std::runtime_error("Index/Range out of bounds:\n" + fs.str());
  }

} // namespace nda
