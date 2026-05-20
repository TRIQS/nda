#include <mpi/mpi.hpp>
#include <nda/mpi.hpp>
#include <nda/nda.hpp>
#include <iostream>

int main(int argc, char **argv) {
  // initialize MPI environment
  mpi::environment env(argc, argv);
  mpi::communicator comm;

  // create a 2x2 array on each process and fill it with its rank
  nda::array<int, 2> A(2, 2);
  A = comm.rank();

  // reduce the array over all processes
  auto A_sum = mpi::reduce(A);

  // print the result
  if (comm.rank() == 0) std::cout << A_sum << std::endl;
}
