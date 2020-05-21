#include "./test_common.hpp"
#include <functional>
#include <nda/array_adapter.hpp>

// ==============================================================

// non default constructible (ndc)
struct ndc {
  int i            = 2;
  ndc(ndc const &) = default;
  ndc(ndc &&)      = default;
  ndc &operator=(ndc const &) = default;
  ndc &operator=(ndc &&) = default;

  ndc(int j) : i(j) { std::cerr << " constructing ndc : " << i << "\n"; }
};

//---------------------

TEST(NDA, NonDefaultConstructible) { //NOLINT

  nda::array<ndc, 2> a(nda::array_adapter{std::array{2, 2}, [](int i, int j) { return i + 10 * j; }});

  nda::array<ndc, 1> a1(nda::array_adapter{std::array{2}, [](int i) { return i; }});

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) { EXPECT_EQ(a(i, j).i, i + 10 * j); }

  //
  auto b = a;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) { EXPECT_EQ(b(i, j).i, i + 10 * j); }

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) b(i, j).i = 0;

  b = a;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) { EXPECT_EQ(b(i, j).i, i + 10 * j); }

  // view is ok
  //auto b = a();

  //a.resize(3,3); // does not compile
}

//---------------------

TEST(NDA, NonDefaultConstructibleInitList1) { //NOLINT

  nda::array<ndc, 1> a{12, 3};

  EXPECT_EQ(a(0).i, 12);
  EXPECT_EQ(a(1).i, 3);
}

//---------------------

TEST(NDA, NonDefaultConstructibleInitList2) { //NOLINT

  nda::array<ndc, 2> a{{12, 3}, {34, 67}};

  EXPECT_EQ(a(0, 0).i, 12);
  EXPECT_EQ(a(0, 1).i, 3);
  EXPECT_EQ(a(1, 0).i, 34);
  EXPECT_EQ(a(1, 1).i, 67);
}

// ==============================================================

// a little non copyable object
struct anc {
  int i            = 2;
  anc()            = default;
  anc(anc const &) = delete;
  anc(anc &&)      = default;
  anc &operator=(anc const &) = delete;
  anc &operator=(anc &&) = default;
};

TEST(NDA, array_of_non_copyable) { //NOLINT
  std::vector<nda::array<anc, 1>> a(2);
  a.emplace_back(2);
}

// ==============================================================

struct S {
  double x = 0, y = 0;
  int i = 0;
  S()   = default;
  S(double x, double y, int k) : x(x), y(y), i(k) {}
};

TEST(NDA, non_numeric1) { //NOLINT

  nda::array<S, 2> A(2, 2);

  S s0{1.0, 2.0, 3};
  int p = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j, ++p) A(i, j) = S{double(i), double(j), p};

  A() = s0;
  p   = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j, ++p) EXPECT_EQ(A(i, j).i, 3);
}

// ---------------------------------------

TEST(NDA, array_of_array) { //NOLINT

  nda::array<nda::array<int, 1>, 2> a(2, 2);
  nda::array<int, 1> a0{1, 2, 3};

  a() = a0;

  EXPECT_EQ(a(1, 0), a0);
}

// ---------------------------------------

TEST(NDA, matrix_of_function) { //NOLINT

  nda::array<std::function<double(double)>, 2> F(2, 2);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      auto s  = i + j;
      F(i, j) = [s](int k) { return k + s; };
    }

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) EXPECT_EQ(F(i, j)(0), i + j);
}

// ==============================================================

MAKE_MAIN
