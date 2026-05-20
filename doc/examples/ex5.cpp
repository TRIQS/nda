#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include <h5/h5.hpp>
#include <iostream>
#include <string>
#include <tuple>

int main(int argc, char *argv[]) {
  // HDF5 file
  h5::file file("ex5.h5", 'w');

  // original array
  auto A = nda::array<int, 2>(5, 5);
  for (int i = 0; auto &x : A) x = i++;
  std::cout << "A = " << A << std::endl;

  // write the array to the file
  h5::write(file, "A", A);

  // write a view to HDF5
  h5::write(file, "A_v", A(nda::range(0, 5, 2), nda::range(0, 5, 2)));

  // read a dataset into an array
  auto A_r = nda::array<int, 2>();
  h5::read(file, "A", A_r);
  std::cout << "A_r = " << A_r << std::endl;

  // read a dataset into a view
  auto B = nda::zeros<int>(5, 5);
  auto B_v = B(nda::range(1, 4), nda::range(0, 5, 2));
  h5::read(file, "A_v", B_v);
  std::cout << "B = " << B << std::endl;

  // prepare a dataset
  h5::write(file, "B", nda::zeros<int>(5, 5));

  // a slice that specifies every other row of "B"
  auto slice_r024 = std::make_tuple(nda::range(0, 5, 2), nda::range::all);

  // write the first 3 rows of A to the slice
  h5::write(file, "B", A(nda::range(0, 3), nda::range::all), slice_r024);

  // write the last 2 rows of A to the empty rows in "B"
  auto slice_r13 = std::make_tuple(nda::range(1, 5, 2), nda::range::all);
  h5::write(file, "B", A(nda::range(3, 5), nda::range::all), slice_r13);

  // read every other row of "A" into the first 3 rows of C
  auto C = nda::zeros<int>(5, 5);
  auto C_r012 = C(nda::range(0, 3), nda::range::all);
  h5::read(file, "A", C_r012, slice_r024);
  std::cout << "C = " << C << std::endl;

  // read rows 1 and 3 of "A" into the empty 2 rows of C
  auto C_r13 = C(nda::range(3, 5), nda::range::all);
  h5::read(file, "A", C_r13, slice_r13);
  std::cout << "C = " << C << std::endl;

  // write an array of strings
  auto S = nda::array<std::string, 1>{"Hi", "my", "name", "is", "John"};
  h5::write(file, "S", S);

  // read an array of strings
  auto S_r = nda::array<std::string, 1>();
  h5::read(file, "S", S_r);
  std::cout << "S_r = " << S_r << std::endl;

  // write an array of integer arrays
  auto I = nda::array<nda::array<int, 1>, 1>{nda::array<int, 1>{0, 1, 2}, nda::array<int, 1>{3, 4, 5}, nda::array<int, 1>{6, 7, 8}};
  h5::write(file, "I", I);

  // read an array of integer arrays
  auto I_r = nda::array<nda::array<int, 1>, 1>();
  h5::read(file, "I", I_r);
  std::cout << "I_r = " << I_r << std::endl;
}
