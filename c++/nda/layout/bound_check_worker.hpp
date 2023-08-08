// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2022 Simons Foundation
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
//
// Authors: Olivier Parcollet, Nils Wentzell

#include "./range.hpp"

namespace nda::details {

  // -------------------- bound_check_worker ---------------------
  //
  // A worker that checks all arguments and gather potential errors in error_code
  struct bound_check_worker {
    long const *lengths{}; // length of input slice
    uint32_t error_code = 0;
    int ellipsis_loss   = 0;
    int N               = 0;

    void f(long key) {
      if ((key < 0) or (key >= lengths[N])) error_code += 1ul << N; // binary code
      ++N;
    }

    void f(range::all_t) { ++N; }
    void f(range const &r) {
      if (r.size() > 0) {
        auto first_idx = r.first();
        auto last_idx  = first_idx + (r.size() - 1) * r.step();
        if (first_idx < 0 or first_idx >= lengths[N] or last_idx < 0 or last_idx >= lengths[N]) error_code += 1ul << N;
      }
      ++N;
    }
    void f(ellipsis) { N += ellipsis_loss + 1; }

    void g(std::stringstream &fs, long key) {
      if (error_code & (1ull << N)) fs << "argument " << N << " = " << key << " is not within [0," << lengths[N] << "[\n";
      N++;
    }
    void g(std::stringstream &, range) { ++N; }
    void g(std::stringstream &, range::all_t) { ++N; }
    void g(std::stringstream &, ellipsis) { N += ellipsis_loss + 1; }
  };

  template <typename... Args>
  void assert_in_bounds(int rank, long const *lengths, Args const &...args) {
    bound_check_worker w{lengths};
    w.ellipsis_loss = rank - sizeof...(Args); // len of ellipsis : how many ranges are missing
    (w.f(args), ...);                         // folding with , operator ...
    if (!w.error_code) return;
    w.N = 0;
    std::stringstream fs;
    (w.g(fs, args), ...); // folding with , operator ...
    //std::cerr  << " key out of domain \n" + fs.str()<<std::endl;
    //EXPECTS_WITH_MESSAGE(false, "Stopping");
    throw std::runtime_error(" key out of domain \n" + fs.str());
  }

} // namespace nda::details
