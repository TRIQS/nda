// Copyright (c) 2020--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#ifdef __GNUC__
// to remove the warning of ; too much. Remove the comma, and clang-format is a mess
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include <iostream>
#include <nda/nda.hpp>
#include <benchmark/benchmark.h>

auto _ = nda::range::all;
nda::ellipsis ___;

using namespace nda;
