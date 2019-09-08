" A special little script to deduce matrix.hxx from array.hxx and same for
" views. The file are quite close, 
" we just register the command to go from one to another
" in order to port quickly any changes
" Section BEGIN_REMOVE_FOR_MATRIX ... END_REMOVE_FOR_MATRIX are removed
" Section UNCOMMENT_FOR_MATRIX : next block is uncommented
"
normal 5GdG
:r c++/nda/array.hpp

:silent

" Remove Rank and replace by 2
:%s/, int Rank//g
:%s/, Rank//g
:%s/== Rank/== 2/g
:%s/Rank, //g
:%s/Rank/2/g

" array-> matrix
" and a few corrections
:%s/array/matrix/g
:%s/is_ndmatrix_v/is_ndarray_v/ge
:%s/IdxMap::rank(),//ge
:%s/std::matrix<long>/std::array<long,2>/ge
:%s/idx_map<Layout>/idx_map<2,Layout>/g

%s/2 == 2 and//g

%s/view\.hpp/view.hxx/g

%s/BEGIN_REMOVE_FOR_MATRIX\_.\{-}END_REMOVE_FOR_MATRIX/DELETED_CODE/g

normal gg
while search("UNCOMMENT_FOR_MATRIX", '', line("$")) > 0 " limit the search to end of file 
       " the last j is crucial or the cursor get up to the block and we have
       " an infinte loop
       normal V},cij

endwhile       

"Clang format
normal ==

