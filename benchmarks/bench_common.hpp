#ifdef __GNUC__
// to remove the warning of ; too much. Remove the comma, and clang-format is a mess
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include <iostream>
#include <nda/nda.hpp>
#include <benchmark/benchmark.h>

nda::range_all _;
nda::ellipsis ___;

using namespace nda;

