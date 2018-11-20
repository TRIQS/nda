/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2017 by the Simons Foundation
 * author : O. Parcollet
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

namespace nda { 

/**
 * for i
 *   for j
 *     for k 
 *     .... 
 *        f(i,j,k ...)
 * */
template<typename F, int R> 
 void for_each(std::array<long, R> idx_lenghts, F && f);

// cheating ...
template<typename F, int R> 
%for R in range(1,10):  
template<typename F> 
  void for_each(std::array<long, R> idx_lenghts, F && f) { 
   %for i in range (R):
     for (int i_${i} =0; i_${i} < idx_lenghts[${i}]; ++i_${i}) 
   %endfor
   F(${', '.join('i_%s'%i for i in range (R))});
  }
%endfor

}
