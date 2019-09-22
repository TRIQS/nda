#define NDA_ENFORCE_BOUNDCHECK
#include "./test_common.hpp"
#include <h5/h5.hpp>

// FIXME  RENAME THIS FILE
#include <nda/h5/simple_read_write.hpp>
//------------------------------------

using nda::range;

template <typename T>
void one_simple(std::string name, T scalar) {

   int N1 = 5, N2 = 7;
  nda::array<T, 2> b(N1, N2), b_sli;

  // numbers are unique ...
  for (int i = 0; i < N1; ++i)
    for (int j = 0; j < N2; ++j) { b(i, j) = scalar*(10 * i + j); }

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

TEST(Slice, Long) { one_simple<long>("long", 1); }

TEST(Slice, Dcomplex) { one_simple<dcomplex>("dcomplex", (1.0 + 1.0i)); }

//------------------------------

TEST(Slice, Systematic3d) {

  int N1 = 3, N2 = 5, N3 = 8;
  int StepMax = 3;
  
  nda::array<long, 3> c(N1, N2, N3), c_sli;

  for (int i = 0; i < N1; ++i)
    for (int j = 0; j < N2; ++j)
      for (int k = 0; k < N3; ++k) { c(i, j, k) = (i + 10 * j + 100 * k); }

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
