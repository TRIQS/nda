/*
 
    Cf mapped_function for vim howto

    : let @a = '"byew"cy$G"tPVG:s/XXX/\=@b/g{VG:s/NNN/\=@c/g'

  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The XXX dimension. Equivalent to a.shape()[NNN]
  /// 
  template <typename A>
  long XXX_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
    return a.shape()[NNN];
  }

first 0
second 1
third 2
fourth 3
fifth 4
sixth 5
seventh 6
eighth 7
ninth 8

  */

// --- generated : do not edit, cf vim macro above ...

/// Access to dimension
/// @tparam A Type modeling NdArray
/// @param a Object
/// @return The first dimension. Equivalent to a.shape()[0]
///
template <typename A>
long first_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
  return a.shape()[0];
}

/// Access to dimension
/// @tparam A Type modeling NdArray
/// @param a Object
/// @return The first dimension. Equivalent to a.shape()[0]
///
template <typename A>
long first_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
  return a.shape()[0];
}

/// Access to dimension
/// @tparam A Type modeling NdArray
/// @param a Object
/// @return The second dimension. Equivalent to a.shape()[1]
///
template <typename A>
long second_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
  return a.shape()[1];
}

/// Access to dimension
/// @tparam A Type modeling NdArray
/// @param a Object
/// @return The third dimension. Equivalent to a.shape()[2]
///
template <typename A>
long third_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
  return a.shape()[2];
}

/// Access to dimension
/// @tparam A Type modeling NdArray
/// @param a Object
/// @return The fourth dimension. Equivalent to a.shape()[3]
///
template <typename A>
long fourth_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
  return a.shape()[3];
}

/// Access to dimension
/// @tparam A Type modeling NdArray
/// @param a Object
/// @return The fifth dimension. Equivalent to a.shape()[4]
///
template <typename A>
long fifth_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
  return a.shape()[4];
}

/// Access to dimension
/// @tparam A Type modeling NdArray
/// @param a Object
/// @return The sixth dimension. Equivalent to a.shape()[5]
///
template <typename A>
long sixth_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
  return a.shape()[5];
}

/// Access to dimension
/// @tparam A Type modeling NdArray
/// @param a Object
/// @return The seventh dimension. Equivalent to a.shape()[6]
///
template <typename A>
long seventh_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
  return a.shape()[6];
}

/// Access to dimension
/// @tparam A Type modeling NdArray
/// @param a Object
/// @return The eighth dimension. Equivalent to a.shape()[7]
///
template <typename A>
long eighth_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
  return a.shape()[7];
}

/// Access to dimension
/// @tparam A Type modeling NdArray
/// @param a Object
/// @return The ninth dimension. Equivalent to a.shape()[8]
///
template <typename A>
long ninth_dim(A const &a) REQUIRES(is_ndarray_v<A>) {
  return a.shape()[8];
}
