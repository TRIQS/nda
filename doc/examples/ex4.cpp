#include <nda/nda.hpp>
#include <array>
#include <iostream>
#include <utility>

int main() {
  // create a full view on an array
  auto A = nda::array<int, 2>(5, 5);
  for (int i = 0; auto &x : A) x = i++;
  auto A_v = A();
  std::cout << "A_v = " << A_v << std::endl;

  // create a view of a view
  auto A_vv = A_v();
  std::cout << "A_vv = " << A_vv << std::endl;

  // check the value type of the view and assign to it
  static_assert(std::is_same_v<decltype(A_v)::value_type, int>);
  A_v(0, 0) = -1;
  std::cout << "A = " << A << std::endl;

  // taking a view of a const array
  auto A_vc = static_cast<const nda::array<int, 2>>(A)();
  static_assert(std::is_same_v<decltype(A_vc)::value_type, const int>);

  // taking a view of a view with a const value type
  auto A_vvc = A_vc();
  static_assert(std::is_same_v<decltype(A_vvc)::value_type, const int>);

  // original array
  A(0, 0) = 0;
  std::cout << "A = " << A << std::endl;

  // slice containing every other column
  auto S_1 = A(nda::range::all, nda::range(0, 5, 2));
  std::cout << "S_1 = " << S_1 << std::endl;

  // slice containing every other row of S_1
  auto S_2 = S_1(nda::range(0, 5, 2), nda::range::all);
  std::cout << "S_2 = " << S_2 << std::endl;

  // slice containing every other row and column
  auto S_3 = A(nda::range(0, 5, 2), nda::range(0, 5, 2));
  std::cout << "S_3 = " << S_3 << std::endl;

  // make a copy of the view using nda::make_regular
  auto A_tmp = nda::make_regular(S_3);

  // assign a scalar to a view
  S_3 = 0;
  std::cout << "S_3 = " << S_3 << std::endl;

  // check the changes to the original array and other views
  std::cout << "S_1 = " << S_1 << std::endl;
  std::cout << "A = " << A << std::endl;

  // assign an array to a view
  S_3 = A_tmp;
  std::cout << "A = " << A << std::endl;

  // copy construct a view
  auto S_3_copy = S_3;
  std::cout << "S_3.data() == S_3_copy.data() = " << (S_3.data() == S_3_copy.data()) << std::endl;

  // move construct a view
  auto S_3_move = std::move(S_3);
  std::cout << "S_3.data() == S_3_move.data() = " << (S_3.data() == S_3_move.data()) << std::endl;

  // copy assign to a view
  auto B = nda::array<int, 2>(S_3.shape());
  auto B_v = B();
  B_v = S_3;
  std::cout << "B = " << B << std::endl;

  // move assign to a view
  auto C = nda::array<int, 2>(S_3.shape());
  nda::array_view<int, 2> C_v = C();
  C_v = std::move(S_3);
  std::cout << "C = " << C << std::endl;

  // arithmetic operations and math functions
  C_v = S_3 * 2 + nda::pow(S_3, 2);
  std::cout << "C = " << C << std::endl;

  // algorithms
  std::cout << "min_element(S_3) = " << nda::min_element(S_3) << std::endl;
  std::cout << "max_element(S_3) = " << nda::max_element(S_3) << std::endl;
  std::cout << "sum(S_3) = " << nda::sum(S_3) << std::endl;

  // rebind a view
  std::cout << "S_3.data() == C_v.data() = " << (S_3.data() == C_v.data()) << std::endl;
  C_v.rebind(S_3);
  std::cout << "S_3.data() == C_v.data() = " << (S_3.data() == C_v.data()) << std::endl;

  // take a view on a std::array
  std::array<double, 5> arr{1.0, 2.0, 3.0, 4.0, 5.0};
  auto arr_v = nda::basic_array_view(arr);
  std::cout << "arr_v = " << arr_v << std::endl;

  // change the value of the vector through the view
  arr_v *= 2.0;
  std::cout << "arr = (";
  for (auto x : arr) std::cout << " " << x;
  std::cout << " )" << std::endl;

  // original array
  auto D = nda::array<int, 2>(3, 4);
  for (int i = 0; auto &x : D) x = i++;
  std::cout << "D = " << D << std::endl;

  // create a transposed view
  auto D_t = nda::transpose(D);
  std::cout << "D_t = " << D_t << std::endl;

  // reshape the array
  auto D_r = nda::reshape(D, 2, 6);
  std::cout << "D_r = " << D_r << std::endl;

  // reshape the view
  auto D_tr = nda::reshape(D_t, 2, 6);
  std::cout << "D_tr = " << D_tr << std::endl;

  // flatten the view
  auto D_t_flat = nda::flatten(D_t);
  std::cout << "D_t_flat = " << D_t_flat << std::endl;

  // flatten the array using reshape
  auto D_flat = nda::reshape(D, D.size());
  std::cout << "D_flat = " << D_flat << std::endl;
}
