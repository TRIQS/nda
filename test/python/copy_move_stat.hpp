// Copyright (c) 2020 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>

#include <nda/basic_array.hpp>

//// Forward Declaration
//struct copy_move_stat;

//// A simple view type
//struct copy_move_stat_view {
  //friend class copy_move_stat;
  //copy_move_stat(copy_move_stat & c): ptr{&c} {}

  //long copy_count() { return ptr->copy_count(); }
  //long move_count() { return ptr->move_count(); }
  //long reset_count() { return ptr->reset_count(); }
  //private:
  //copy_move_stat * ptr;
//}

// Class to track numbers of copy/move
struct copy_move_stat {
  copy_move_stat(bool verbose = true) : verbose(verbose) {
    if (verbose) std::cout << "Construction\n";
    ++construction_count;
  }

  copy_move_stat(copy_move_stat const & c) : verbose(c.verbose) {
    if (verbose) std::cout << "Copy Construction\n";
    ++copy_construction_count;
  }

  copy_move_stat(copy_move_stat && c) : verbose(c.verbose){
    if (verbose) std::cout << "Move Construction\n";
    ++move_construction_count;
  }

  //copy_move_stat(copy_move_stat_view v): copy_move_stat{v->ptr} {}

  copy_move_stat &operator=(copy_move_stat const & c) {
    verbose = c.verbose;
    if (verbose) std::cout << "Copy Assignment\n";
    ++copy_assignment_count;
    return *this;
  }

  copy_move_stat &operator=(copy_move_stat && c) {
    verbose = c.verbose;
    if (verbose) std::cout << "Move Assignment\n";
    ++move_assignment_count;
    return *this;
  }

  ~copy_move_stat() {
    if (verbose) std::cout << "Destruction\n";
    ++destruction_count;
  }

  inline static long construction_count      = 0;
  inline static long copy_construction_count = 0;
  inline static long move_construction_count = 0;
  inline static long copy_assignment_count   = 0;
  inline static long move_assignment_count   = 0;
  inline static long destruction_count       = 0;

  static long copy_count() { return copy_construction_count + copy_assignment_count; }

  static long move_count() { return move_construction_count + move_assignment_count; }

  static void reset() {
    std::cout << "Resetting counters .. \n\n\n";
    copy_construction_count = 0;
    move_construction_count = 0;
    copy_assignment_count   = 0;
    move_assignment_count   = 0;
    destruction_count       = 0;
  }

  private:
  bool verbose;
};

struct member_stat {
  member_stat() = default;
  copy_move_stat m = {};
};

copy_move_stat make_obj() {
  return {};
}

nda::array<copy_move_stat, 1> make_arr(long n) {
  auto shape = std::array{n};
  return nda::array<copy_move_stat, 1>{shape};
}

nda::array<copy_move_stat, 2> make_arr(long n1, long n2) {
  auto shape = std::array{n1, n2};
  return nda::array<copy_move_stat, 2>{shape};
}

long take_obj(copy_move_stat o) {
  return o.copy_count() + o.move_count();
}

long take_arr(nda::array<copy_move_stat, 1> const & a) {
  NDA_PRINT(a.shape());
  return a[0].copy_count() + a[0].move_count();
}

long take_arr(nda::array<copy_move_stat, 2> const & a) {
  return a(0,0).copy_count() + a(0,0).move_count();
}

//void take_arr_v(nda::array<copy_move_stat_view, 1> a) {}
