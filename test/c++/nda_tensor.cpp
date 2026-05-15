// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <complex>

using namespace std::complex_literals;
using nda::C_layout, nda::F_layout;
using nda::mem::Host, nda::mem::Device, nda::mem::Unified;

TEST(NDA, TensorDefaultIndices) {
  EXPECT_EQ(nda::tensor::default_index<0>(), "");
  EXPECT_EQ(nda::tensor::default_index<1>(), "a");
  EXPECT_EQ(nda::tensor::default_index<2>(), "ab");
  EXPECT_EQ(nda::tensor::default_index<5>(), "abcde");
  EXPECT_EQ(nda::tensor::default_index<10>(), "abcdefghij");
  EXPECT_EQ(nda::tensor::default_index<26>(), "abcdefghijklmnopqrstuvwxyz");
}
