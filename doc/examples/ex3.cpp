#include <nda/nda.hpp>
#include <complex>
#include <iostream>

int main() {
  using namespace std::complex_literals;

  // assign a scalar to an array
  auto A = nda::array<std::complex<double>, 2>(3, 2);
  A = 0.1 + 0.2i;
  std::cout << "A = " << A << std::endl;

  // assign a scalar to a matrix
  auto M = nda::matrix<std::complex<double>>(3, 2);
  M = 0.1 + 0.2i;
  std::cout << "M = " << M << std::endl;

  // assign an array to an array
  auto A_arr = nda::array<nda::array<int, 1>, 1>(4);
  nda::array<int, 1> A_sub{1, 2, 3};
  for (auto &x : A_arr) x = A_sub;
  std::cout << "A_arr = " << A_arr << std::endl;

  // copy assignment
  auto M_copy = nda::matrix<std::complex<double>>(3, 2);
  M_copy = M;
  std::cout << "M_copy = " << M_copy << std::endl;

  // move assignment
  auto M_move = nda::matrix<std::complex<double>>();
  M_move = std::move(M_copy);
  std::cout << "M_move = " << M_move << std::endl;
  std::cout << "M_copy.empty() = " << M_copy.empty() << std::endl;

  // assign a lazy expression
  nda::matrix<std::complex<double>, nda::F_layout> M_f(3, 2);
  M_f = M + M;
  std::cout << "M_f = " << M_f << std::endl;

  // assign another array with a different layout and algebra
  nda::array<std::complex<double>, 2> A2;
  A2 = M_f;
  std::cout << "A2 = " << A2 << std::endl;

  // assign a contiguous range to an 1-dimensional array
  std::vector<long> vec{1, 2, 3, 4, 5};
  auto A_vec = nda::array<long, 1>();
  A_vec = vec;
  std::cout << "A_vec = " << A_vec << std::endl;

  // initialize an array using traditional for-loops
  auto B = nda::array<int, 2>(2, 3);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      B(i, j) = i * 3 + j;
    }
  }
  std::cout << "B = " << B << std::endl;

  // initialize an array using a range-based for-loop
  auto B2 = nda::array<int, 2>(2, 3);
  for (int i = 0; auto &x : B2) x = i++;
  std::cout << "B2 = " << B2 << std::endl;

  // initialize an array using nda::for_each
  auto B3 = nda::array<int, 2>(2, 3);
  nda::for_each(B3.shape(), [&B3](auto i, auto j) { B3(i, j) = i * 3 + j; });
  std::cout << "B3 = " << B3 << std::endl;

  // initialize an array using CLEF's automatic assignment
  using namespace nda::clef::literals;
  auto C = nda::array<int, 4>(2, 2, 2, 2);
  C(i_, j_, k_, l_) << i_ * 8 + j_ * 4 + k_ * 2 + l_;
  std::cout << "C = " << C << std::endl;
}
