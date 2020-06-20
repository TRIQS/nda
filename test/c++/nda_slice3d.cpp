#include "./test_common.hpp"

// additional tests on slice
// Some are already in basic, and view

// isolate : a bit long to compile with sanitizer
// Check the calculation of layout_prop in a 4d slices

template <auto StrideOrder>
void test3d() {

  std::cerr << " ======  " << nda::decode<3>(StrideOrder) << "==========" << std::endl;
  int N0 = 3, N1 = 4, N2 = 5;
  range R0(1, N0 - 1);
  range R1(1, N1 - 1);
  range R2(1, N2 - 1);

  using A = nda::array<long, 3, nda::basic_layout<0, StrideOrder, nda::layout_prop_e::contiguous>>;
  A a(N0, N1, N2);

  {
    long c = 0;
    for (auto &x : a) {
      // Checks that the iterator in indeed going one by one in memory, even with a non trivial StrideOrder
      EXPECT_EQ(std::distance(&(*std::begin(a)), &x), c);
      x = c++; // in memory order, 0,1,2,3,4 ....
    }
  }

  //PRINT(a);

  auto check = [n = 0](auto v) mutable {
    //PRINT(v);
    //PRINT(v.indexmap());

    bool is_contiguous          = (v.indexmap().layout_prop == nda::layout_prop_e::contiguous);
    bool is_strided_1d          = (has_strided_1d(v.indexmap().layout_prop));
    bool smallest_stride_is_one = (has_smallest_stride_is_one(v.indexmap().layout_prop));

    EXPECT_EQ(is_contiguous, (is_strided_1d and smallest_stride_is_one));

    std::cerr << "n = " << n << " contiguous " << is_contiguous << " is_strided_1d = " << is_strided_1d
              << " smallest_stride_is_one = " << smallest_stride_is_one << std::endl;

    bool check_strided_1d = true;

    // forcing the basic iterator in rank dimension, avoiding all optimization
    using layout_t = typename decltype(v)::layout_t;
    using Iterator = nda::array_iterator<layout_t::rank(), long const, typename nda::default_accessor::template accessor<long>::pointer>;
    auto it        = Iterator{nda::permutations::apply(layout_t::stride_order, v.indexmap().lengths()),
                       nda::permutations::apply(layout_t::stride_order, v.indexmap().strides()), v.data_start(), false};
    auto end       = Iterator{nda::permutations::apply(layout_t::stride_order, v.indexmap().lengths()),
                        nda::permutations::apply(layout_t::stride_order, v.indexmap().strides()), v.data_start(), true};

    auto it1 = it;
    auto c   = *it1; // first element
    ++it1;
    auto stri = (*it1) - c; // the stride
    for (; it != end; ++it) {
      check_strided_1d &= (*it == c);
      c += stri;
    }

    bool check_contiguous = check_strided_1d and (stri == 1);

    EXPECT_EQ(check_strided_1d, is_strided_1d);
    EXPECT_EQ(check_contiguous, is_contiguous);
    EXPECT_EQ(v.indexmap().is_contiguous(), is_contiguous);
    EXPECT_EQ(v.indexmap().is_strided_1d(), is_strided_1d);

    ++n;
  };

  check(a(_, _, _));
  check(a(_, _, 2));

  check(a(_, 2, _));
  check(a(_, 2, 2));

  check(a(2, _, _));
  check(a(2, _, 2));
  check(a(2, 2, _));
  //check(a(2, 2, 2));
}

// --------------  Check the calculation of layout_prop in a 4d slices
TEST(Slice, ContiguityComputation3dfull) { //NOLINT

  test3d<nda::encode(std::array{0, 1, 2})>();
  test3d<nda::encode(std::array{1, 0, 2})>();
  test3d<nda::encode(std::array{2, 0, 1})>();
  test3d<nda::encode(std::array{0, 2, 1})>();
  test3d<nda::encode(std::array{1, 2, 0})>();
  test3d<nda::encode(std::array{2, 1, 0})>();
}
