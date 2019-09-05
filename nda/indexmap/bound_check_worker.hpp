
namespace nda::details {

  class my_runtime_error : public std::exception {
    std::stringstream acc;
    std::string _trace;
    mutable std::string _what;

    public:
    my_runtime_error() throw() : std::exception() {} // _trace = }//utility::stack_trace(); }
    my_runtime_error(my_runtime_error const &e) throw() : acc(e.acc.str()), _trace(e._trace), _what(e._what) {}
    virtual ~my_runtime_error() throw() {}
    template <typename T>
    my_runtime_error &operator<<(T const &x) {
      acc << x;
      return *this;
    }
    my_runtime_error &operator<<(const char *mess) {
      (*this) << std::string(mess);
      return *this;
    } // to limit code size
    virtual const char *what() const throw() {
      std::stringstream out;
      out << acc.str() << "\n.. Error occurred on node ";
      //if (mpi::is_initialized()) out << mpi::communicator().rank() << "\n";
      if (getenv("TRIQS_SHOW_EXCEPTION_TRACE")) out << ".. C++ trace is : " << trace() << "\n";
      _what = out.str();
      return _what.c_str();
    }
    virtual const char *trace() const throw() { return _trace.c_str(); }
  };

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
    void g(std::stringstream &, range) { ++N; }
    void g(std::stringstream &, range_all) { ++N; }
    void g(std::stringstream &, ellipsis) { N += ellipsis_loss + 1; }
  };

  template <typename... Args>
  void assert_in_bounds(int rank, long const *lengths, Args const &... args) {
    bound_check_worker w{lengths};
    w.ellipsis_loss = rank - sizeof...(Args); // len of ellipsis : how many ranges are missing
    (w.f(args), ...);                         // folding with , operator ...
    if (!w.error_code) return;
    w.N = 0;
    std::stringstream fs;
    (w.g(fs, args), ...); // folding with , operator ...
    std::string s = " key out of domain \n" + fs.str();
    throw my_runtime_error() << "EEE"; //s.c_str());
  }

} // namespace nda::details
