#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include <nda/linalg.hpp>
#include <h5/h5.hpp>
#include <iostream>

int main() {
  // create a 3x2 integer array and initialize it
  auto A = nda::array<int, 2>(3, 2);
  for (int i = 0; auto &x : A) x = i++;

  // create a 3x2 integer array using initializer lists
  auto A2 = nda::array<int, 2>{{0, 1}, {2, 3}, {4, 5}};

  // create a 3x2 integer array in Fortran order and initialize it
  auto B = nda::array<int, 2, nda::F_layout>(3, 2);
  for (int i = 0; auto &x : B) x = i++;

  // print the formatted arrays, its sizes and its shapes
  std::cout << "A = " << A << std::endl;
  std::cout << "A.size() = " << A.size() << std::endl;
  std::cout << "A.shape() = " << A.shape() << std::endl;
  std::cout << std::endl;
  std::cout << "B = " << B << std::endl;
  std::cout << "B.size() = " << B.size() << std::endl;
  std::cout << "B.shape() = " << B.shape() << std::endl;

  // access and assign to a single element
  A(2, 1) = 100;
  B(2, 1) = A(2, 1);
  std::cout << "A = " << A << std::endl;
  std::cout << "B = " << B << std::endl;
  std::cout << "B(2, 1) = " << B(2, 1) << std::endl;

  // assigning a scalar to an array
  A = 42;
  std::cout << "A = " << A << std::endl;

  // assigning an array to another array
  A = B;
  std::cout << "A = " << A << std::endl;

  // create a view on A
  auto A_v = A();
  std::cout << "A_v =" << A_v << std::endl;
  std::cout << "A_v.size() = " << A_v.size() << std::endl;
  std::cout << "A_v.shape() = " << A_v.shape() << std::endl;

  // manipulate the data in the view
  A_v(0, 1) = -12;
  std::cout << "A = " << A << std::endl;

  // create a slice on A (a view on its second row)
  auto A_s = A(1, nda::range::all);
  std::cout << "A_s = " << A_s << std::endl;
  std::cout << "A_s.size() = " << A_s.size() << std::endl;
  std::cout << "A_s.shape() = " << A_s.shape() << std::endl;

  // assign to a slice
  A_s = nda::array<int, 1>{-1, -4};
  std::cout << "A_s = " << A_s << std::endl;
  std::cout << "A = " << A << std::endl;

  // add two arrays element-wise
  auto C = nda::array<int, 2>({{1, 2}, {3, 4}});
  auto D = nda::array<int, 2>({{5, 6}, {7, 8}});
  std::cout << C + D << std::endl;

  // evaluate the lazy expression by assigning it to an existing array
  auto E = nda::array<int, 2>(2, 2);
  E = C + D;
  std::cout << "E = " << E << std::endl;

  // evaluate the lazy expression by constructing a new array
  std::cout << "nda::array<int, 2>{C + D} = " << nda::array<int, 2>{C + D} << std::endl;

  // evaluate the lazy expression using nda::make_regular
  std::cout << "nda::make_regular(C + D) = " << nda::make_regular(C + D) << std::endl;

  // multiply two arrays elementwise
  std::cout << "C * D =" << nda::array<int, 2>(C * D) << std::endl;

  // multiply two matrices
  auto M1 = nda::matrix<int>({{1, 2}, {3, 4}});
  auto M2 = nda::matrix<int>({{5, 6}, {7, 8}});
  std::cout << "M1 * M2 =" << nda::matrix<int>(M1 * M2) << std::endl;

  // element-wise square
  nda::array<double, 2> C_sq = nda::pow(C, 2);
  std::cout << "C^2 =" << C_sq << std::endl;

  // element-wise square root
  nda::array<double, 2> C_sq_sqrt = nda::sqrt(C_sq);
  std::cout << "sqrt(C^2) =" << C_sq_sqrt << std::endl;

  // trace of a matrix
  std::cout << "trace(M1) = " << nda::trace(M1) << std::endl;

  // find the minimum/maximum element in an array
  std::cout << "min_element(C) =" << nda::min_element(C) << std::endl;
  std::cout << "max_element(C) =" << nda::max_element(C) << std::endl;

  // is any (are all) element(s) greater than 1?
  auto greater1 = nda::map([](int x) { return x > 1; })(C);
  std::cout << "any(C > 1) = " << nda::any(greater1) << std::endl;
  std::cout << "all(C > 1) = " << nda::all(greater1) << std::endl;

  // sum/multiply all elements in an array
  std::cout << "sum(C) = " << nda::sum(C) << std::endl;
  std::cout << "product(C) = " << nda::product(C) << std::endl;

  // write an array to an HDF5 file
  h5::file out_file("ex1.h5", 'w');
  h5::write(out_file, "C", C);

  // read an array from an HDF5 file
  nda::array<int, 2> C_copy;
  h5::file in_file("ex1.h5", 'r');
  h5::read(in_file, "C", C_copy);
  std::cout << "C_copy = " << C_copy << std::endl;

  // create a 2x2 matrix
  auto M3 = nda::matrix<double>{{1, 2}, {3, 4}};

  // get the inverse of the matrix (calls LAPACK routines)
  auto M3_inv = nda::linalg::inv(M3);
  std::cout << "M3_inv = " << M3_inv << std::endl;

  // get the inverse of a matrix manually using its adjugate and determinant
  auto M3_adj = nda::matrix<double>{{4, -2}, {-3, 1}};
  auto M3_det = nda::linalg::det(M3);
  auto M3_inv2 = nda::matrix<double>(M3_adj / M3_det);
  std::cout << "M3_inv2 = " << M3_inv2 << std::endl;

  // check the inverse using matrix-matrix multiplication
  std::cout << "M3 * M3_inv = " << M3 * M3_inv << std::endl;
  std::cout << "M3_inv * M3 = " << M3_inv * M3 << std::endl;

  // solve a linear system using the inverse
  auto b = nda::vector<double>{5, 8};
  auto x = M3_inv * b;
  std::cout << "M3 * x = " << M3 * x << std::endl;

  // initialize a 4x7 array using CLEF automatic assignment
  using namespace nda::clef::literals;
  auto F = nda::array<int, 2>(4, 7);
  F(i_, j_) << i_ * 7 + j_;
  std::cout << "F = " << F << std::endl;
}
