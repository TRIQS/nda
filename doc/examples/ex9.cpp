#include <nda/nda.hpp>
#include <complex>
#include <iostream>

int main() {
  using nda::tensor::binary_op;
  using nda::tensor::unary_op;

  // construct a rank-3 tensor of shape 2x3x4 and fill it with consecutive integers
  auto A = nda::array<double, 3>(2, 3, 4);
  for (int n = 0; auto &x : A) x = n++;
  std::cout << "A = " << A << std::endl;
  std::cout << "A.shape() = " << A.shape() << std::endl;

  // default index strings: one letter per rank, starting from 'a'
  std::cout << "default_index<3>() = " << nda::tensor::default_index<3>() << std::endl;

  // in-place tensor addition: B_ijk <- alpha * A_ijk + beta * B_ijk
  auto B    = nda::array<double, 3>(2, 3, 4);
  B         = 1.0;
  double a1 = 2.0;
  double b1 = 3.0;
  nda::tensor::add(a1, A, "ijk", b1, B, "ijk");
  std::cout << "B = 2*A + 3 = " << B << std::endl;

  // tensor contraction: C_il = sum_jk A_ijk * B_jkl
  auto A2 = nda::array<double, 3>(2, 3, 4);
  for (int n = 0; auto &x : A2) x = 0.1 * n++;
  auto B2 = nda::array<double, 3>(3, 4, 5);
  for (int n = 0; auto &x : B2) x = 0.1 * n++;
  auto C2 = nda::array<double, 2>::zeros({2, 5});
  nda::tensor::contract(1.0, A2, "ijk", B2, "jkl", 0.0, C2, "il");
  std::cout << "C_il = A_ijk * B_jkl = " << C2 << std::endl;

  // matrix-matrix multiplication expressed as a rank-2 contraction: C_ik = A_ij * B_jk
  auto Am = nda::array<double, 2>{{1, 2, 3}, {4, 5, 6}};
  auto Bm = nda::array<double, 2>{{7, 8}, {9, 10}, {11, 12}};
  auto Cm = nda::array<double, 2>::zeros({2, 2});
  nda::tensor::contract(1.0, Am, "ij", Bm, "jk", 0.0, Cm, "ik");
  std::cout << "A_ij * B_jk = " << Cm << std::endl;

  // tensor dot product (sum over all matching indices, returns a scalar)
  auto d_indexed = nda::tensor::dot(A2, "ijk", A2, "ijk");
  auto d_default = nda::tensor::dot(A2, A2);
  std::cout << "dot(A2, A2) [indexed] = " << d_indexed << std::endl;
  std::cout << "dot(A2, A2) [default] = " << d_default << std::endl;

  // full tensor reductions to a scalar with different binary operations
  std::cout << "reduce(A, SUM)    = " << nda::tensor::reduce(A) << std::endl;
  std::cout << "reduce(A, MAX)    = " << nda::tensor::reduce(A, binary_op::MAX) << std::endl;
  std::cout << "reduce(A, MIN)    = " << nda::tensor::reduce(A, binary_op::MIN) << std::endl;
  std::cout << "reduce(A, NORM_2) = " << nda::tensor::reduce(A, binary_op::NORM_2) << std::endl;

  // in-place tensor scaling with an optional element-wise unary operation
  auto S = nda::array<double, 3>(2, 3, 4);
  for (int n = 0; auto &x : S) x = static_cast<double>(n++ + 1);
  nda::tensor::scale(2.0, S);
  std::cout << "2 * S = " << S << std::endl;
  nda::tensor::scale(1.0, S, unary_op::SQRT);
  std::cout << "sqrt(2 * S) = " << S << std::endl;

  // CONJ acts on a complex tensor
  using cplx = std::complex<double>;
  auto Z     = nda::array<cplx, 3>(2, 2, 2);
  for (int n = 0; auto &x : Z) {
    x = cplx(n, n + 1);
    ++n;
  }
  nda::tensor::scale(cplx{1.0, 0.0}, Z, unary_op::CONJ);
  std::cout << "conj(Z) = " << Z << std::endl;

  // element-wise binary operation: B_ijk <- op(alpha * A_ijk, beta * B_ijk)
  auto E1 = nda::array<double, 3>(2, 3, 4);
  for (int n = 0; auto &x : E1) x = static_cast<double>(n++);
  auto E2 = nda::array<double, 3>(2, 3, 4);
  E2      = 1.0;
  nda::tensor::elementwise(1.0, E1, "ijk", 1.0, E2, "ijk", binary_op::SUM);
  std::cout << "E2 <- E1 + E2 = " << E2 << std::endl;

  auto F1 = nda::array<double, 3>(2, 3, 4);
  F1      = 2.0;
  auto F2 = nda::array<double, 3>(2, 3, 4);
  F2      = 3.0;
  nda::tensor::elementwise(1.0, F1, "ijk", 1.0, F2, "ijk", binary_op::PROD);
  std::cout << "F2 <- F1 * F2 = " << F2 << std::endl;
}
