// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_common.hpp"

#include <nda/gtest_tools.hpp>
#include <nda/nda.hpp>

#include <array>
#include <type_traits>
#include <vector>
#include <tuple>

#include <nda/h5.hpp>
//#include <hdf5_hl.h>

#include <H5Tpublic.h> // HDF5 type creation API

struct S {
  int a;
  double x;
  int u                                                         = 1;
  friend constexpr bool operator<=>(const S &lhs, const S &rhs) = default;
};

template <>
h5::hid_t h5::detail::hid_t_of<S>() {
  hid_t type = H5Tcreate(H5T_COMPOUND, sizeof(S));
  H5Tinsert(type, "s", HOFFSET(S, a), H5T_NATIVE_INT);
  H5Tinsert(type, "x", HOFFSET(S, x), H5T_NATIVE_DOUBLE);
  H5Tinsert(type, "u", HOFFSET(S, u), H5T_NATIVE_INT);
  return type;
}

template <>
constexpr bool h5::is_h5_compound<S> = true;

TEST(NDA, H5Compound) {
  nda::array<S, 1> A{{1, 2.0}, {3, 4.0}, {5, 6.0}, {7, 8.0}, {9, 10.0}}; // NOLINT
  S s{1, 1.3, -1};                                                       // NOLINT
  static_assert(requires { h5::detail::hid_t_of<decltype(A)::value_type>(); });
  {
    h5::file out("compound.h5", 'w');
    h5::write(out, "A", A);
    h5::write(out, "s", s);
  }

  {
    h5::file in("compound.h5", 'r');
    auto B  = h5::read<nda::array<S, 1>>(in, "A");
    auto sb = h5::read<S>(in, "s");
    EXPECT_EQ(A, B);
    EXPECT_EQ(s, sb);
  }
};