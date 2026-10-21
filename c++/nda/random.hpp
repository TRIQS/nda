// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#pragma once

#include <random>

namespace nda::random {

  inline std::mt19937_64 &engine() {
    static std::mt19937_64 gen{};
    return gen;
  }

  inline void set_seed(std::mt19937_64::result_type value) { engine().seed(value); }

}
