// Copyright (c) 2019-2021 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once

#include <gtest/gtest.h>

#include <array>
#include <numeric>

#ifndef NDA_DEBUG
#define NDA_DEBUG
#endif // NDA_DEBUG

#ifndef NDA_ENFORCE_BOUNDCHECK
#define NDA_ENFORCE_BOUNDCHECK
#endif // NDA_ENFORCE_BOUNDCHECK

// Check if function arguments are equal.
template <typename T, typename... Ts>
bool are_equal(const T &a, const Ts &...args) {
  return ((a == args) && ...);
}

// Type of the shape array.
template <std::size_t R>
using shape_t = std::array<long, R>;

// Non trivial, default constructible type.
struct foo_def_ctor {
  int x{10};
  foo_def_ctor() = default;
  foo_def_ctor(int x) : x(x) {}
  foo_def_ctor(const foo_def_ctor &)            = default;
  foo_def_ctor(foo_def_ctor &&)                 = default;
  foo_def_ctor &operator=(const foo_def_ctor &) = default;
  foo_def_ctor &operator=(foo_def_ctor &&)      = default;
  foo_def_ctor &operator=(int y) {
    x = y;
    return *this;
  };
  bool operator==(const foo_def_ctor &f) const { return x == f.x; }
};

// Non trivial, non default constructible type.
struct foo_non_def_ctor {
  int x{10};
  foo_non_def_ctor(int x) : x(x) {}
  foo_non_def_ctor(const foo_non_def_ctor &)            = default;
  foo_non_def_ctor(foo_non_def_ctor &&)                 = default;
  foo_non_def_ctor &operator=(const foo_non_def_ctor &) = default;
  foo_non_def_ctor &operator=(foo_non_def_ctor &&)      = default;
  foo_non_def_ctor &operator=(int y) {
    x = y;
    return *this;
  };
  bool operator==(const foo_non_def_ctor &f) const { return x == f.x; }
};

// Non trivial, non copyable type.
struct foo_non_copy {
  int x{10};
  foo_non_copy() = default;
  foo_non_copy(int x) : x(x) {}
  foo_non_copy(const foo_non_copy &)            = delete;
  foo_non_copy(foo_non_copy &&)                 = default;
  foo_non_copy &operator=(const foo_non_copy &) = delete;
  foo_non_copy &operator=(foo_non_copy &&)      = default;
  foo_non_copy &operator=(int y) {
    x = y;
    return *this;
  };
  bool operator==(const foo_non_copy &f) const { return x == f.x; }
};

// Satisfies the nda::ArrayInitializer concept.
template <typename T, std::size_t R>
struct array_initializer {
  using value_type = T;
  std::array<long, R> shape_;
  array_initializer(const std::array<long, R> &shape) : shape_(shape) {}
  [[nodiscard]] auto shape() const { return shape_; }
  [[nodiscard]] auto size() const { return std::accumulate(shape_.begin(), shape_.end(), 1l, std::multiplies<>{}); }
  void invoke(auto &&A) const {
    int i = 0;
    for (auto &x : A) x = static_cast<value_type>(i++);
  }
};

// Satisfies the nda::ArrayOfRank<R> concept.
template <typename T, std::size_t R>
struct array_of_rank {
  using value_type = T;
  std::array<long, R> shape_;
  array_of_rank(const std::array<long, R> &shape) : shape_(shape) {}
  [[nodiscard]] auto shape() const { return shape_; }
  [[nodiscard]] auto size() const { return std::accumulate(shape_.begin(), shape_.end(), 1l, std::multiplies<>{}); }
  [[nodiscard]] auto operator()(auto &&...) const { return static_cast<value_type>(R); }
};
