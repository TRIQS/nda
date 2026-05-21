#include <nda/nda.hpp>
#include <nda/sym_grp.hpp>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

int main() {
  // size of the matrix
  constexpr int N = 3;

  // some useful typedefs
  using idx_t = std::array<long, 2>;
  using sym_t = std::tuple<idx_t, nda::operation>;
  using sym_func_t = std::function<sym_t(idx_t const &)>;

  // define a hermitian symmetry (satisfies nda::NdaSymmetry)
  auto h_symmetry = [](idx_t const &x) {
    if (x[0] == x[1]) return sym_t{x, nda::operation{false, false}}; // sign flip = false, complex conjugate = false
    return sym_t{idx_t{x[1], x[0]}, nda::operation{false, true}};    // sign flip = false, complex conjugate = true
  };

  // construct the symmetry group
  nda::array<std::complex<double>, 2> A(N, N);
  auto grp = nda::sym_grp{A, std::vector<sym_func_t>{h_symmetry}};

  // check the number of symmetry classes
  std::cout << "Number of symmetry classes: " << grp.num_classes() << std::endl;

  // print the symmetry classes
  for (int i = 1; auto const &c : grp.get_sym_classes()) {
    std::cout << "Symmetry class " << i << ":" << std::endl;
    for (auto const &x : c) {
      std::cout << "  Idx: " << x.first << ", Sign flip: " << x.second.sgn << ", Complex conjugation: " << x.second.cc << std::endl;
    }
    ++i;
  }

  // print flat indices
  nda::array<int, 2> B(3, 3);
  for (int i = 0; auto &x : B) x = i++;
  std::cout << B << std::endl;

  // define an initializer function
  int count_eval = 0;
  auto init_func = [&count_eval, &B](idx_t const &idx) {
    ++count_eval;
    const double val = B(idx[0], idx[1]);
    return std::complex<double>{val, val};
  };

  // initialize the array using the symmetry group
  grp.init(A, init_func);
  std::cout << "A = " << A << std::endl;

  // check the number of evaluations
  std::cout << "Number of evaluations: " << count_eval << std::endl;

  // get representative elements
  auto reps = grp.get_representative_data(A);
  auto reps_view = nda::array_view<std::complex<double>, 1>(reps);
  std::cout << "Representative elements = " << reps_view << std::endl;

  // use representative data to initialize a new array
  reps_view *= 2.0;
  nda::array<std::complex<double>, 2> B_sym(N, N);
  grp.init_from_representative_data(B_sym, reps);
  std::cout << "B_sym = " << B_sym << std::endl;

  // symmetrize an already symmetric array
  auto v1 = grp.symmetrize(A);
  std::cout << "Symmetrized A = " << A << std::endl;
  std::cout << "Max. violation at index " << v1.second << " = " << v1.first << std::endl;

  // change an off-diagonal element
  A(0, 2) *= 2.0;
  std::cout << "A = " << A << std::endl;

  // symmetrize again
  auto v2 = grp.symmetrize(A);
  std::cout << "Symmetrized A = " << A << std::endl;
  std::cout << "Max. violation at index " << v2.second << " = " << v2.first << std::endl;
}
