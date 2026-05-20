#include <nda/nda.hpp>
#include <nda/mpi.hpp>
#include <mpi/mpi.hpp>
#include <iostream>

template <typename A>
void print(const A &arr, const mpi::communicator &comm) {
  std::cout << "Rank " << comm.rank() << ": " << arr << std::endl;
}

int main(int argc, char *argv[]) {
  // initialize MPI environment and communicator
  mpi::environment env(argc, argv);
  mpi::communicator comm;
  const int root = 0;

  // create an array and initialize it on root
  auto A = nda::array<int, 2, nda::F_layout>();
  if (comm.rank() == root) {
    A.resize(2, comm.size());
    for (int i = 0; auto &x : A) x = i++;
  }
  print(A, comm);
  comm.barrier();

  // broadcast the array from root to all other ranks
  mpi::broadcast(A, comm);
  print(A, comm);
  comm.barrier();

  // prepare the array
  A = 0;
  if (comm.rank() == root) {
    A(nda::range::all, root) = 1;
  }
  print(A, comm);
  comm.barrier();

  // broadcast the first column from root to all other ranks
  auto A_v = A(nda::range::all, comm.rank());
  mpi::broadcast(A_v, comm);
  print(A, comm);
  comm.barrier();

  // prepare the array to be gathered
  auto B = nda::array<int, 1>(comm.rank() + 1);
  B() = comm.rank();
  print(B, comm);
  comm.barrier();

  // gather the arrays on root
  auto B_g = mpi::gather(B, comm, root);
  print(B_g, comm);
  comm.barrier();

  // all-gather the arrays
  auto B_g_all = mpi::all_gather(B, comm);
  (void)B_g_all;

  // resize and reshape the arrays and gather the resulting views
  B.resize(4);
  B = comm.rank();
  auto B_r = nda::reshape(B, 2, 2);
  auto B_rg = mpi::gather(B_r, comm, root);
  print(B_rg, comm);
  comm.barrier();

  // gather Fortran-layout arrays
  auto B_f = nda::array<int, 2, nda::F_layout>(2, 2);
  B_f = comm.rank();
  auto B_fg = mpi::gather(nda::transpose(B_f), comm, root);
  print(B_fg, comm);
  comm.barrier();

  // transpose the result
  print(nda::transpose(B_fg), comm);
  comm.barrier();

  // scatter an array from root to all other ranks
  nda::array<int, 2> C = mpi::scatter(B_rg, comm, root);
  print(C, comm);
  comm.barrier();

  // scatter an array with extents not divisible by the number of ranks
  auto C_s = mpi::scatter(C, comm, 2);
  print(C_s, comm);
  comm.barrier();

  // reduce an array on root using MPI_SUM
  auto D = mpi::reduce(C, comm, root);
  print(D, comm);
  comm.barrier();

  // all-reduce an array in-place
  mpi::all_reduce_in_place(C, comm);
  print(C, comm);
  comm.barrier();

  // scatter a view into an existing array
  auto C_into = nda::array<int, 2>(2, 2);
  mpi::scatter_into(B_rg, C_into, comm, root);
  print(C_into, comm);
  comm.barrier();
}
