/*
    File is generated by vim.
   To regenerate the file,
     1- load the vim function by (or add this to your .vim ?)
        :source c++/nda/vimexpand.vim
     2- all it
      call VimExpand2Simple()
  
  VIMEXPAND2 first,0 second,1 third,2 fourth,3 fifth,4 sixth,5 seventh,6 eighth,7 ninth,8
  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The @1 dimension. Equivalent to a.shape()[@2]
  /// 
  template <Array A>
  [[deprecated("@1_dim is deprecated (convergence to std::mdspan). Replace by .extent(@2)")]] 
  long @1_dim(A const &a) {
    return a.shape()[@2];
  }

*/

namespace nda {

  // --- VIMEXPAND_START  generated : do not edit, cf vim macro above ...


  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The first dimension. Equivalent to a.shape()[0]
  /// 
  template <Array A>
  [[deprecated("first_dim is deprecated (convergence to std::mdspan). Replace by .extent(0)")]] 
  long first_dim(A const &a) {
    return a.shape()[0];
  }

  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The second dimension. Equivalent to a.shape()[1]
  /// 
  template <Array A>
  [[deprecated("second_dim is deprecated (convergence to std::mdspan). Replace by .extent(1)")]] 
  long second_dim(A const &a) {
    return a.shape()[1];
  }

  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The third dimension. Equivalent to a.shape()[2]
  /// 
  template <Array A>
  [[deprecated("third_dim is deprecated (convergence to std::mdspan). Replace by .extent(2)")]] 
  long third_dim(A const &a) {
    return a.shape()[2];
  }

  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The fourth dimension. Equivalent to a.shape()[3]
  /// 
  template <Array A>
  [[deprecated("fourth_dim is deprecated (convergence to std::mdspan). Replace by .extent(3)")]] 
  long fourth_dim(A const &a) {
    return a.shape()[3];
  }

  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The fifth dimension. Equivalent to a.shape()[4]
  /// 
  template <Array A>
  [[deprecated("fifth_dim is deprecated (convergence to std::mdspan). Replace by .extent(4)")]] 
  long fifth_dim(A const &a) {
    return a.shape()[4];
  }

  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The sixth dimension. Equivalent to a.shape()[5]
  /// 
  template <Array A>
  [[deprecated("sixth_dim is deprecated (convergence to std::mdspan). Replace by .extent(5)")]] 
  long sixth_dim(A const &a) {
    return a.shape()[5];
  }

  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The seventh dimension. Equivalent to a.shape()[6]
  /// 
  template <Array A>
  [[deprecated("seventh_dim is deprecated (convergence to std::mdspan). Replace by .extent(6)")]] 
  long seventh_dim(A const &a) {
    return a.shape()[6];
  }

  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The eighth dimension. Equivalent to a.shape()[7]
  /// 
  template <Array A>
  [[deprecated("eighth_dim is deprecated (convergence to std::mdspan). Replace by .extent(7)")]] 
  long eighth_dim(A const &a) {
    return a.shape()[7];
  }

  /// Access to dimension
  /// @tparam A Type modeling NdArray
  /// @param a Object
  /// @return The ninth dimension. Equivalent to a.shape()[8]
  /// 
  template <Array A>
  [[deprecated("ninth_dim is deprecated (convergence to std::mdspan). Replace by .extent(8)")]] 
  long ninth_dim(A const &a) {
    return a.shape()[8];
  }


}
