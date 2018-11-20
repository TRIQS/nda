/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011-2017 by O. Parcollet
 * Copyright (C) 2018 by Simons Foundation
 *   author : O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include <vector>

namespace nda::mem {

  // -------------- ref count table -----------------------

  // A simple table of counters to count the references to a memory block (handle, cf below).
  // The counters are allocated and freed along the principle of a freelist allocator.
  class rtable_t {
    public:
    using int_t = int32_t;
    std::vector<int_t> ta; // table of the counters
    long nac = 1;          // position of the next available counter.
                           // We dont use 0. (id =0 will mean "no counter" in the handle)
                           // possible optimization : nac == ta[0].

    void release(int_t p) { // the counter p in not in use
      ta[p] = nac;          // freelist advance by 1 :
      nac   = p;
      std::cerr << "released " << p << " nac = " << nac << std::endl;
    }

    public:
    rtable_t(long size = 10) : ta(size) {
      for (long u = 0; u < ta.size(); ++u) ta[u] = u + 1; // 1, 2, 3 ...
    }

    ~rtable_t() {
      if (!empty()) std::cerr << "Error detected in reference counting ! No all refs have been rereferenced\n";
    }

    // For checking purpose only. Check that the chain of nac is a complete permutation of the vector
    bool empty() const {
      std::vector<bool> m(ta.size(), true);
      m[0]    = false;
      long nc = nac;
      for (long u = 0; (u < ta.size()) and (nc < ta.size()); ++u) { // u is here to break possible infinite looo
        std::cerr << nc << "->";
        m[nc] = false;
        nc    = ta[nc];
      }
      std::cerr << nc << std::endl;
      long s = 0;
      for (auto x : m) {
        std::cerr << x << std::endl;
        s += (x ? 1 : 0);
      }
      return s == 0;
    }

    // yield the number of a new counter >0, and set the counter to 1.
    long get() {
      long r = nac;
      nac    = ta[nac];
      ta[r]  = 1;
      if (nac == ta.size()) { ta.push_back(ta.size() + 1); }
      std::cerr << "got " << r << " nac = " << nac << std::endl;
      return r;
    }

    // access to the refs
    std::vector<int_t> const &nrefs() const { return ta; }

    // increase the counter number p
    void incref(long p) { ++ta[p]; }

    // decrease the counter number p. Return true iif it has reached 0.
    // If it has reached 0, it also releases the counter.
    bool decref(long p) {
      --ta[p];
      std::cerr << "decref " << p << " nref : " << ta[p] << "  nac = " << nac << std::endl;
      bool reached0 = (ta[p] == 0);
      if (reached0) release(p);
      return reached0;
    }
  };
} // namespace nda::mem
