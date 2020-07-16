#include <nda/nda.hpp>
#include <nda/print.hpp>

struct member_access {
  nda::array<long, 1> arr = {1, 2, 3};
  nda::array<nda::array<long, 1>, 1> arr_arr = {{1, 2, 3}, {1, 2}};
};

nda::array<long, 1> make_arr(long n) {
  auto res = nda::array<long, 1>{std::array{n}};
  for(int i = 0; i < n; ++i){
    res[i] = i;
  }
  return res;
}

nda::array<long, 2> make_arr(long n1, long n2) {
  auto res = nda::array<long, 2>{std::array{n1, n2}};
  for(int i = 0; i < n1; ++i){
    for(int j = 0; j < n2; ++j){
      res(i,j) = j + i * n2;
    }
  }
  return res;
}

nda::array<nda::array<long, 1>,1> make_arr_arr(long n1, long n2) {
  auto res = nda::array<nda::array<long, 1>, 1>{std::array{n1}};
  for(int i = 0; i < n1; ++i){
    res(i) = nda::array<long, 1>{std::array{n2}};
    for(int j = 0; j < n2; ++j){
      res(i)(j) = j + i * n2;
    }
  }
  return res;
}

// =================== C2PY ===================

long size_arr(nda::array<long, 1> const & a) { return a.size(); }
long size_arr(nda::array<long, 2> const & a) { return a.size(); }

long size_arr_v(nda::array_view<long, 1> a) { return a.size(); }
long size_arr_v(nda::array_view<long, 2> a) { return a.size(); }

long size_arr_cv(nda::array_const_view<long, 1> a) { return a.size(); }
long size_arr_cv(nda::array_const_view<long, 2> a) { return a.size(); }

long size_arr_arr(nda::array<nda::array<long, 1>, 1> a) { return a.size(); }
long size_arr_arr_v(nda::array<nda::array_view<long, 1>, 1> a) { return a.size(); }
long size_arr_arr_cv(nda::array<nda::array_const_view<long, 1>, 1> a) { return a.size(); }

// =================== PY2C + C2PY ===================

//auto ret_arr(nda::array<long, 1> const & a) { return a; }
//auto ret_arr(nda::array<long, 2> const & a) { return a; }

//auto ret_arr_v(nda::array_view<long, 1> a) { return a; }
//auto ret_arr_v(nda::array_view<long, 2> a) { return a; }

//auto ret_arr_cv(nda::array_const_view<long, 1> a) { return a; }
//auto ret_arr_cv(nda::array_const_view<long, 2> a) { return a; }

//auto ret_arr_arr(nda::array<nda::array<long, 1>, 1> a) { return a; }
//auto ret_arr_arr_v(nda::array<nda::array_view<long, 1>, 1> a) { return a; }
//auto ret_arr_arr_cv(nda::array<nda::array_const_view<long, 1>, 1> a) { return a; }
