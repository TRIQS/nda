// Copyright (c) 2023--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

#include "./test_sym_grp.hpp"

#include <nda/gtest_tools.hpp>

TEST(NDAMpi, SymGrpMatrixPermutation) {
  test_sym_grp_matrix_permutation(false);
  test_sym_grp_matrix_permutation(true);
}

TEST(NDAMpi, SymGrpMatrixFlipShift) {
  test_sym_grp_matrix_flip_shift(false);
  test_sym_grp_matrix_flip_shift(true);
}

TEST(NDAMpi, SymGrpTensorCylicTriplet) {
  test_sym_grp_tensor_cyclic_triplet(false);
  test_sym_grp_tensor_cyclic_triplet(true);
}

MPI_TEST_MAIN
