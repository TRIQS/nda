#include <nda/nda.hpp>
#include <nda/linalg.hpp>
#include <nda/lapack.hpp>
#include <complex>
#include <iostream>

int main() {
  // construct a 4x3 matrix and a vector of size 3
  nda::matrix<double> M(4, 3);
  nda::vector<double> v(3);

  // construct a 3x3 identity matrix
  auto I = nda::eye<double>(3);
  std::cout << "I = " << I << std::endl;

  // construct a 2x2 diagonal matrix
  auto D = nda::diag(nda::array<int, 1>{1, 2});
  std::cout << "D = " << D << std::endl;

  // assigning a scalar to a matrix and vector
  M = 3.1415;
  v = 2.7182;
  std::cout << "M = " << M << std::endl;
  std::cout << "v = " << v << std::endl;

  // get a vector view of the diagonal of a matrix
  auto d = nda::diagonal(D);
  std::cout << "d = " << d << std::endl;
  std::cout << "Algebra of d: " << nda::get_algebra<decltype(d)> << std::endl;

  // transform an array into a matrix view
  auto A = nda::array<int, 2>{{1, 2}, {3, 4}};
  auto A_mv = nda::make_matrix_view(A);
  std::cout << "A_mv = " << A_mv << std::endl;
  std::cout << "Algebra of A: " << nda::get_algebra<decltype(A)> << std::endl;
  std::cout << "Algebra of A_mv: " << nda::get_algebra<decltype(A_mv)> << std::endl;

  // take the cross product of two vectors
  auto v1 = nda::vector<double>{1.0, 2.0, 3.0};
  auto v2 = nda::vector<double>{4.0, 5.0, 6.0};
  auto v3 = nda::linalg::cross_product(v1, v2);
  std::cout << "v1 = " << v1 << std::endl;
  std::cout << "v2 = " << v2 << std::endl;
  std::cout << "v3 = v1 x v2 = " << v3 << std::endl;

  // check the cross product using the dot product
  std::cout << "v1 . v3 = " << nda::linalg::dot(v1, v3) << std::endl;
  std::cout << "v2 . v3 = " << nda::linalg::dot(v2, v3) << std::endl;

  // cross product via matrix-vector product
  auto M_v1 = nda::matrix<double>{{0, -v1[2], v1[1]}, {v1[2], 0, -v1[0]}, {-v1[1], v1[0], 0}};
  std::cout << "M_v1 = " << M_v1 << std::endl;
  auto v3_mv = M_v1 * v2;
  std::cout << "v3_mv = " << v3_mv << std::endl;

  // define a symmetric matrix
  auto M1 = nda::matrix<double>{{4, -14, -12}, {-14, 10, 13}, {-12, 13, 1}};
  std::cout << "M1 = " << M1 << std::endl;

  // calculate the eigenvalues and eigenvectors of a symmetric matrix
  auto [s, Q] = nda::linalg::eigh(M1);
  std::cout << "Eigenvalues of M1: s = " << s << std::endl;
  std::cout << "Eigenvectors of M1: Q = " << Q << std::endl;

  // check the eigendecomposition
  auto M1_reconstructed = Q * nda::diag(s) * nda::transpose(Q);
  std::cout << "M1_reconstructed = " << M1_reconstructed << std::endl;

  // define the linear system of equations and the right hand side matrix
  auto A1 = nda::matrix<double, nda::F_layout>{{3, 2, -1}, {2, -2, 4}, {-1, 0.5, -1}};
  auto b1 = nda::vector<double>{1, -2, 0};
  std::cout << "A1 = " << A1 << std::endl;
  std::cout << "b1 = " << b1 << std::endl;

  // LU factorization using the LAPACK interface
  auto ipiv = nda::vector<int>(3);
  auto LU = A1;
  auto info = nda::lapack::getrf(LU, ipiv);
  if (info != 0) {
    std::cerr << "Error: nda::lapack::getrf failed with error code " << info << std::endl;
    return 1;
  }

  // solve the linear system of equations using the LU factorization
  nda::matrix<double, nda::F_layout> x1(3, 1);
  x1(nda::range::all, 0) = b1;
  info = nda::lapack::getrs(LU, x1, ipiv);
  if (info != 0) {
    std::cerr << "Error: nda::lapack::getrs failed with error code " << info << std::endl;
    return 1;
  }
  std::cout << "x1 = " << x1(nda::range::all, 0) << std::endl;

  // check the solution
  std::cout << "A1 * x1 = " << A1 * x1(nda::range::all, 0) << std::endl;
}
