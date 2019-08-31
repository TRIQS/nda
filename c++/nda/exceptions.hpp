
/*******************************************************************************
 *
 * NDA: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2011 by M. Ferrero, O. Parcollet
 *
 * NDA is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * NDA is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * NDA. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once

//#include "../mpi/base.hpp"
//#include "./stack_trace.hpp"
#include <exception>
#include <string>
#include <sstream>
#include <stdlib.h>

#define NDA_RUNTIME_ERROR throw nda::runtime_error{} << "Error at " << __FILE__ << " : " << __LINE__ << "\n\n"

/*#define NDA_ASSERT(X)                                                                                                                                \*/
//if (!(X)) NDA_RUNTIME_ERROR << BOOST_PP_STRINGIZE(X);
//#define NDA_ASSERT2(X, ...)                                                                                                                          \
  //if (!(X)) NDA_RUNTIME_ERROR << BOOST_PP_STRINGIZE(X) << "\n " << __VA_ARGS__;

namespace nda {

  class runtime_error : public std::exception {
    std::stringstream acc;
    std::string _trace;
    mutable std::string _what;

    public:
    runtime_error() noexcept : std::exception() {} // _trace = utility::stack_trace(); }

    runtime_error(runtime_error const &e) noexcept : acc(e.acc.str()), _trace(e._trace), _what(e._what) {}

    virtual ~runtime_error() noexcept {}

    template <typename T>
    runtime_error &operator<<(T const &x) {
      acc << x;
      return *this;
    }

    runtime_error &operator<<(const char *mess) {
      (*this) << std::string(mess);
      return *this;
    } // to limit code size

    virtual const char *what() const noexcept {
      std::stringstream out;
      out << acc.str() << "\n.. Error occurred on node ";
      //if (mpi::is_initialized()) out << mpi::communicator().rank() << "\n";
      //if (getenv("NDA_SHOW_EXCEPTION_TRACE")) out << ".. C++ trace is : " << trace() << "\n";
      _what = out.str();
      return _what.c_str();
    }

    //virtual const char *trace() const noexcept { return _trace.c_str(); }
  };
} // namespace nda
