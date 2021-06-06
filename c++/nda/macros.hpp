// Copyright (c) 2019-2020 Simons Foundation
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

#ifndef _CCQ_MACROS_GUARD_H
#define _CCQ_MACROS_GUARD_H

// CCQ, TRIQS general macros
// GUARD IT do not use pragma once
// hence one can simply include them in every projects

// --- Stringify macros -----

#define AS_STRING(...) AS_STRING2(__VA_ARGS__)
#define AS_STRING2(...) #__VA_ARGS__

#define PRINT(X) std::cerr << AS_STRING(X) << " = " << X << "      at " << __FILE__ << ":" << __LINE__ << '\n'
#define NDA_PRINT(X) std::cerr << AS_STRING(X) << " = " << X << "      at " << __FILE__ << ":" << __LINE__ << '\n'

// --- Concept macros -----


#define 
#define REQUIRES requires
#define REQUIRES20 requires

// C++20 explicit(bool) : degrade it NOTHING in c++17, we can not check easily
#define EXPLICIT explicit

// WARNING : it is critical for our doctools to have REQUIRES as requires, NOT a (...) with __VA_ARGS__
// It is the same effect, but raises unnecessary complications in traversing the AST in libtooling with macros.


#endif

// -----------------------------------------------------------

#define FORCEINLINE __inline__ __attribute__((always_inline))

#ifdef NDEBUG

#define EXPECTS(X)
#define ASSERT(X)
#define ENSURES(X)
#define EXPECTS_WITH_MESSAGE(X, ...)
#define ASSERT_WITH_MESSAGE(X, ...)
#define ENSURES_WITH_MESSAGE(X, ...)

#else

#include <iostream>

#define EXPECTS(X)                                                                                                                                   \
  if (!(X)) {                                                                                                                                        \
    std::cerr << "Precondition " << AS_STRING(X) << " violated at " << __FILE__ << ":" << __LINE__ << "\n";                                          \
    std::terminate();                                                                                                                                \
  }
#define ASSERT(X)                                                                                                                                    \
  if (!(X)) {                                                                                                                                        \
    std::cerr << "Assertion " << AS_STRING(X) << " violated at " << __FILE__ << ":" << __LINE__ << "\n";                                             \
    std::terminate();                                                                                                                                \
  }
#define ENSURES(X)                                                                                                                                   \
  if (!(X)) {                                                                                                                                        \
    std::cerr << "Postcondition " << AS_STRING(X) << " violated at " << __FILE__ << ":" << __LINE__ << "\n";                                         \
    std::terminate();                                                                                                                                \
  }

#define EXPECTS_WITH_MESSAGE(X, ...)                                                                                                                 \
  if (!(X)) {                                                                                                                                        \
    std::cerr << "Precondition " << AS_STRING(X) << " violated at " << __FILE__ << ":" << __LINE__ << "\n";                                          \
    std::cerr << "Error message : " << __VA_ARGS__ << std::endl;                                                                                     \
    std::terminate();                                                                                                                                \
  }
#define ASSERT_WITH_MESSAGE(X, ...)                                                                                                                  \
  if (!(X)) {                                                                                                                                        \
    std::cerr << "Assertion " << AS_STRING(X) << " violated at " << __FILE__ << ":" << __LINE__ << "\n";                                             \
    std::cerr << "Error message : " << __VA_ARGS__ << std::endl;                                                                                     \
    std::terminate();                                                                                                                                \
  }
#define ENSURES_WITH_MESSAGE(X, ...)                                                                                                                 \
  if (!(X)) {                                                                                                                                        \
    std::cerr << "Postcondition " << AS_STRING(X) << " violated at " << __FILE__ << ":" << __LINE__ << "\n";                                         \
    std::cerr << "Error message : " << __VA_ARGS__ << std::endl;                                                                                     \
    std::terminate();                                                                                                                                \
  }

#endif

#endif
