#include <nda/nda.hpp>
#include <nda/sym_grp.hpp>

#include <array>
#include <complex>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <tuple>
#include <vector>

int main() {
  constexpr int N = 3;

  // typedefs for the symmetry group
  using idx_t      = std::array<long, 2>;
  using sym_t      = std::tuple<idx_t, nda::operation>;
  using sym_func_t = std::function<sym_t(idx_t const &)>;

  // create an hermitian matrix
  auto A = nda::array<std::complex<double>, 2>::rand(N, N);
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) A(i, j) = std::conj(A(j, i));
  }

  // hermitian symmetry (satisfies nda::NdaSymmetry)
  auto h_symmetry = [](idx_t const &x) {
    idx_t xp = {x[1], x[0]};
    return sym_t{xp, nda::operation{false, true}}; // sign flip = false, complex conjugate = true
  };

  // construct the symmetry group
  auto grp = nda::sym_grp{A, std::vector<sym_func_t>{h_symmetry}};

  // create an initializer function (satisfies nda::NdaInitFunc)
  auto init_func = [&A](idx_t const &x) { return std::apply(A, x); };

  // initialize a second array using the symmetry group and array A
  nda::array<std::complex<double>, 2> B(N, N);
  grp.init(B, init_func);

  // output A and B
  std::cout << A << std::endl;
  std::cout << B << std::endl;

  // get representative data (should be a vector of size 6)
  auto vec = grp.get_representative_data(A);
  std::cout << "\n" << nda::array_view<std::complex<double>, 1>(vec) << std::endl;
}
