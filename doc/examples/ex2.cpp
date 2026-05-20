#include <nda/nda.hpp>
#include <complex>
#include <iostream>

int main(int argc, char *argv[]) {
  // default constructor
  auto A1 = nda::array<double, 2>();
  std::cout << "A1 = " << A1 << std::endl;
  std::cout << "A1.size() = " << A1.size() << std::endl;
  std::cout << "A1.shape() = " << A1.shape() << std::endl;

  // resize the array using the resize method
  A1.resize(3, 2);
  std::cout << "A1.size() = " << A1.size() << std::endl;
  std::cout << "A1.shape() = " << A1.shape() << std::endl;

  // resize using assignment
  A1 = nda::array<double, 2>(10, 10);
  std::cout << "A1.size() = " << A1.size() << std::endl;
  std::cout << "A1.shape() = " << A1.shape() << std::endl;

  // create a 100x100 complex matrix
  auto M1 = nda::matrix<std::complex<double>>(100, 100);
  std::cout << "M1.size() = " << M1.size() << std::endl;
  std::cout << "M1.shape() = " << M1.shape() << std::endl;

  // create a 1-dimensional vector of size 5 with the value 10 + 1i
  using namespace std::complex_literals;
  auto v1 = nda::vector<std::complex<double>>(5, 10 + 1i);
  std::cout << "v1 = " << v1 << std::endl;
  std::cout << "v1.size() = " << v1.size() << std::endl;
  std::cout << "v1.shape() = " << v1.shape() << std::endl;

  // copy constructor
  auto v2 = v1;
  std::cout << "v2 = " << v2 << std::endl;
  std::cout << "v2.size() = " << v2.size() << std::endl;
  std::cout << "v2.shape() = " << v2.shape() << std::endl;

  // move constructor
  auto v3 = std::move(v2);
  std::cout << "v3 = " << v3 << std::endl;
  std::cout << "v3.size() = " << v3.size() << std::endl;
  std::cout << "v3.shape() = " << v3.shape() << std::endl;
  std::cout << "v2.empty() = " << v2.empty() << std::endl;

  // 1-dimensional array from a std::initializer_list
  auto A1_il = nda::array<int, 1>{1, 2, 3, 4, 5};
  std::cout << "A1_il = " << A1_il << std::endl;
  std::cout << "A1_il.size() = " << A1_il.size() << std::endl;
  std::cout << "A1_il.shape() = " << A1_il.shape() << std::endl;

  // 2-dimensional array from a std::initializer_list
  auto A2_il = nda::array<int, 2>{{1, 2}, {3, 4}, {5, 6}};
  std::cout << "A2_il = " << A2_il << std::endl;
  std::cout << "A2_il.size() = " << A2_il.size() << std::endl;
  std::cout << "A2_il.shape() = " << A2_il.shape() << std::endl;

  // 3-dimensional array from a std::initializer_list
  auto A3_il = nda::array<int, 3>{{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}};
  std::cout << "A3_il = " << A3_il << std::endl;
  std::cout << "A3_il.size() = " << A3_il.size() << std::endl;
  std::cout << "A3_il.shape() = " << A3_il.shape() << std::endl;

  // construct an array from a lazy expression
  nda::array<int, 1> A1_sum = A1_il + A1_il;
  std::cout << "A1_sum = " << A1_sum << std::endl;
  std::cout << "A1_sum.size() = " << A1_sum.size() << std::endl;
  std::cout << "A1_sum.shape() = " << A1_sum.shape() << std::endl;

  // construct an array from another array with a different memory layout
  nda::array<double, 2, nda::F_layout> A2_f(A2_il);
  std::cout << "A2_f = " << A2_f << std::endl;
  std::cout << "A2_f.size() = " << A2_f.size() << std::endl;
  std::cout << "A2_f.shape() = " << A2_f.shape() << std::endl;

  // 1-dimensional array with only even integers from 2 to 20
  auto v4 = nda::arange(2, 20, 2);
  std::cout << "v4 = " << v4 << std::endl;

  // 3x3 identity matrix
  auto I = nda::eye<double>(3);
  std::cout << "I = " << I << std::endl;

  // 2x2x2 array with random numbers from 0 to 1
  auto R = nda::rand(2, 2, 2);
  std::cout << "R = " << R << std::endl;
  std::cout << "R.shape() = " << R.shape() << std::endl;

  // 3x6 array with zeros
  auto Z = nda::zeros<double>(3, 6);
  std::cout << "Z = " << Z << std::endl;
}
