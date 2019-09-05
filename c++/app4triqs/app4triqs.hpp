#pragma once
#include <triqs/gfs.hpp>

namespace app4triqs {

  /**
   * A very useful and important class
   *
   * @note A Useful note
   * @include app4triqs/app4triqs.hpp
   */
  class toto {

    int i = 0;

    public:
    toto() = default;

    /**
     * Construct from integer
     *
     * @param i_ a scalar  :math:`G(\tau)`
     */
    explicit toto(int i_) : i(i_) {}

    ~toto() = default;

    // Copy/Move construction
    toto(toto const &) = default;
    toto(toto &&)      = default;

    /// Copy/Move assignment
    toto &operator=(toto const &) = default;
    toto &operator=(toto &&) = default;

    /// Simple accessor
    int get_i() const { return i; }

    /** 
     * A simple function with :math:`G(\tau)`
     *
     * @param u Nothing useful
     */
    int f(int u) { return u; }

    /// Arithmetic operations
    toto operator+(toto const &b) const;
    toto &operator+=(toto const &b);

    /// Comparison
    bool operator==(toto const &b) const;

    /// Serialization
    template <class Archive> void serialize(Archive &ar, const unsigned int) { ar &i; }
  };

  /**
   * Chain digits of two integers
   *
   * @head A set of functions that implement chaining
   *
   * @tail Do I really need to explain more ? 
   *
   * @param i The first integer
   * @param j The second integer
   * @return An integer containing the digits of both i and j
   *
   * @remark
   */
  int chain(int i, int j);

} // namespace app4triqs
