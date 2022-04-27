// Copyright (c) 2019-2020 Simons Foundation
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

#include "./test_common.hpp"
#include <h5/h5.hpp>
#include <nda/h5.hpp>
#include <nda/clef/literals.hpp>

using namespace nda::clef::literals;

template <typename T>
void one_simple(std::string name, T scalar) {

  int N1 = 5, N2 = 7;
  nda::array<T, 2> b(N1, N2), b_sli;

  // numbers are unique ...
  b(i_, j_) << scalar * (10 * i_ + j_);

  std::cout << b << std::endl;
  std::string filename = "ess_slice_simple_" + name + ".h5";

  {
    h5::file file(filename, 'w');
    h5_write(file, "slice", b(_, range(1, 3)));
    h5_write(file, "slice2", b(range(1, 5, 2), range(1, 3)));
    h5_write(file, "slice3", b(range(1, 5, 2), range(0, 7, 2)));
  }

  //b = 0; // to be sure it really tests...

  // READ the file
  {
    h5::file file(filename, 'r');

    h5_read(file, "slice", b_sli);
    EXPECT_EQ_ARRAY(b_sli, b(_, range(1, 3)));

    h5_read(file, "slice2", b_sli);
    EXPECT_EQ_ARRAY(b_sli, b(range(1, 5, 2), range(1, 3)));

    h5_read(file, "slice3", b_sli);
    EXPECT_EQ_ARRAY(b_sli, b(range(1, 5, 2), range(0, 7, 2)));
  }
}
//------------------------------

TEST(SliceH5, Long) { //NOLINT
  one_simple<long>("long", 1);
}

TEST(SliceH5, Dcomplex) { //NOLINT
  one_simple<dcomplex>("dcomplex", (1.0 + 1.0i));
}

//------------------------------

TEST(SliceH5, Systematic3d) { //NOLINT

  int N1 = 3, N2 = 5, N3 = 8;
  int StepMax = 3;

  nda::array<long, 3> c(N1, N2, N3), c_sli;
  c(i_, j_, k_) << (i_ + 10 * j_ + 100 * k_);

  std::cerr << c << std::endl;
  std::string filename = "ess_slice_systematic3d.h5";

  std::cerr << "Writing all slices to disk : may take a few seconds..." << std::endl;

  {
    h5::file file(filename, 'w');

    int count = 0;

    for (int i = 0; i < N1; ++i)
      for (int j = 0; j < N2; ++j)
        for (int k = 0; k < N3; ++k)
          for (int i2 = i + 1; i2 < N1; ++i2)
            for (int j2 = j + 1; j2 < N2; ++j2)
              for (int k2 = k + 1; k2 < N3; ++k2)
                for (int si = 1; si <= StepMax; ++si)
                  for (int sj = 1; sj <= StepMax; ++sj)
                    for (int sk = 1; sk <= StepMax; ++sk) {
                      h5_write(file, "slice" + std::to_string(count++), c(range(i, i2, si), range(j, j2, sj), range(k, k2, sk)));
                    }
  }

  //b = 0; // to be sure it really tests...

  std::cerr << "Rereading all slices from disk ..." << std::endl;
  // READ the file
  {
    int count = 0;
    h5::file file(filename, 'r');
    for (int i = 0; i < N1; ++i)
      for (int j = 0; j < N2; ++j)
        for (int k = 0; k < N3; ++k)
          for (int i2 = i + 1; i2 < N1; ++i2)
            for (int j2 = j + 1; j2 < N2; ++j2)
              for (int k2 = k + 1; k2 < N3; ++k2)
                for (int si = 1; si <= StepMax; ++si)
                  for (int sj = 1; sj <= StepMax; ++sj)
                    for (int sk = 1; sk <= StepMax; ++sk) {
                      h5_read(file, "slice" + std::to_string(count++), c_sli);
                      EXPECT_EQ_ARRAY(c_sli, c(range(i, i2, si), range(j, j2, sj), range(k, k2, sk)));
                    }
  }
}

// ==============================================================

TEST(SliceH5, Hyperslab) { //NOLINT

  nda::array<long, 3> A(4, 5, 6);
  A(i_, j_, k_) << i_ * 100 + j_ * 10 + k_;

  // write to file
  {
    h5::file f("test_nda_slab.h5", 'w');
    h5_write(f, "A", A);
  }

  // auto slice = std::tuple{range(), range(), 3}); // FIXME range() = range(0,-1,0) <- Empty!
  auto slice          = std::tuple{_, range(3, 5), 3};
  auto slice_ellipsis = std::tuple{___, range(3, 5), 3};
  auto A_slice        = A(_, range(3, 5), 3);

  // read only slice of the data
  nda::array<long, 2> B{}, C{};
  {
    // Provide multi-dimensional slice, i.e. tuple of long, range, range_all or ellipsis
    h5::file f("test_nda_slab.h5", 'r');

    h5_read(f, "A", B, slice);
    EXPECT_EQ_ARRAY(A_slice, B);

    h5_read(f, "A", C, slice_ellipsis);
    EXPECT_EQ_ARRAY(A_slice, C);
  }

  // read into non-contiguous view
  auto D = nda::zeros<long>(4, 5, 6);
  auto E = nda::zeros<long>(4, 5, 6);
  {
    h5::file f("test_nda_slab.h5", 'r');

    auto D_slice = D(_, range(3, 5), 3);
    h5_read(f, "A", D_slice, slice);
    EXPECT_EQ_ARRAY(A_slice, D_slice);

    auto E_slice = E(_, range(3, 5), 3);
    h5_read(f, "A", E_slice, slice_ellipsis);
    EXPECT_EQ_ARRAY(A_slice, E_slice);
  }

  // test ellipsis of width two and three
  auto F = nda::array<long, 2>{};
  auto G = nda::array<long, 3>{};
  {
    h5::file f("test_nda_slab.h5", 'r');

    h5_read(f, "A", F, std::tuple{___, 3});
    EXPECT_EQ_ARRAY(A(___, 3), F);

    h5_read(f, "A", G, std::tuple{___});
    EXPECT_EQ_ARRAY(A, G);
  }

  // Update only a slice of the data in file
  A_slice() = 42;
  {
    h5::file f("test_nda_slab.h5", 'a');
    h5_write(f, "A", A_slice, slice);
  }
  auto A2 = nda::array<long, 3>{};
  {
    h5::file f("test_nda_slab.h5", 'r');
    h5_read(f, "A", A2);
  }
  EXPECT_EQ_ARRAY(A, A2);
}
