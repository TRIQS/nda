#include <cstddef>
#include <type_traits>
#include <cstring>

#include "./rtable.hpp"

namespace nda::mem {

  // The global table of reference counters
  rtable_t globals::rtable;

} // namespace nda::mem
