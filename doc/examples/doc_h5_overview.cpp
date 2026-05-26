#include <nda/h5.hpp>
#include <nda/nda.hpp>
#include <h5/h5.hpp>

int main() {
  // create a 4x2 array with random values
  auto A = nda::rand(4, 2);

  // write the array to an HDF5 file
  h5::file file("A.h5", 'w');
  h5::write(file, "A", A);
}
