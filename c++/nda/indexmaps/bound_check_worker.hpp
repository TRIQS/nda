
namespace nda::details {

  // -------------------- bound_check_worker ---------------------
  //
  // A worker that checks all arguments and gather potential errors in error_code
  struct bound_check_worker {
    long const *lengths; // length of input slice
    uint32_t error_code = 0;
    int ellipsis_loss   = 0;
    int N               = 0;

    void f(long key) {
      bool pb = ((key < 0) or (key >= lengths[N]));
      if (pb) error_code += 1ul << N; // binary code
      ++N;
    }

    void f(range_tag) { ++N; }
    void f(ellipsis) { N += ellipsis_loss + 1; }

    void g(std::stringstream &fs, long key) {
      if (error_code & (1ull << N)) fs << "argument " << N << " = " << key << " is not within [0," << lengths[N] << "[\n";
      N++;
    }
    void g(std::stringstream &fs, range) { ++N; }
    void g(std::stringstream &fs, range_all) { ++N; }
    void g(std::stringstream &fs, ellipsis) { N += ellipsis_loss + 1; }
  };

  template <typename... Args> void assert_in_bounds(int rank, long const *lengths, Args const &... args) {
    bound_check_worker w{lengths};
    w.ellipsis_loss = rank - sizeof...(Args); // len of ellipsis : how many ranges are missing
    (w.f(args), ...);                         // folding with , operator ...
    if (!w.error_code) return;
    w.N = 0;
    std::stringstream fs;
    (w.g(fs, args), ...); // folding with , operator ...
    NDA_KEY_ERROR << " key out of domain \n" << fs.str();
  }

} // namespace nda::details
