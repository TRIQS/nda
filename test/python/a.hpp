#include <nda/basic_array.hpp>
#include <nda/map.hpp>

nda::array<long, 1> arrn(int n) {
  nda::array<long, 1> result(n);
  for (int i = 0; i < n; ++i) result(i) = i + 1;
  return result;
}

struct container {
  nda::array<long, 1> a;
  container(int n) { a = arrn(n); }
  nda::array_view<long, 1> get() { return a; }
  // nda::array_view<long const, 1> get_c() const { return a; } FIXME not wrappable
};

nda::array<container, 2> make_container_array(){
  nda::array<int, 2> arr_int{{0, 1}, {2, 3}};
  return nda::map([](int i){ return container(i); })(arr_int);
}
