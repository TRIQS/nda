#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include <h5/h5.hpp>

int main() {
  // create an array of shape (4,4,4)
  nda::array<long, 3> A(4, 4, 4);

  // create an array given its data
  nda::array<long, 2> B{{1, 2}, {3, 4}, {5, 6}};

  // assign a scalar to the full array or a single element
  A()        = 0;
  A(0, 1, 2) = 40;

  // access single elements
  [[maybe_unused]] long a = A(0, 1, 2) + B(0, 1);

  // access a slice of the array of shape (3, 2)
  auto V = A(nda::range(0, 3), nda::range(0, 2), 0);

  // lazy arithmetic operations
  auto C                  = V + 2 * B;            // C is an expression
  [[maybe_unused]] auto D = nda::make_regular(C); // D is an array

  // various algorithms
  nda::min_element(V);
  nda::max_element(V);
  nda::sum(V);

  // write to HDF5 file
  {
    h5::file file("dat.h5", 'w');
    h5::write(file, "A", A);
  }

  // read from HDF5 file
  nda::array<long, 3> E;
  {
    h5::file file("dat.h5", 'r');
    h5::read(file, "A", E);
  }
}
