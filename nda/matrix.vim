normal 6GdG
:r array.hpp

:silent

" Remove Rank and replace by 2
:%s/, int Rank//g
:%s/, Rank//g
:%s/== Rank/== 2/g
:%s/Rank, //g
:%s/Rank/2/g

" 
:%s/array/matrix/g

%s/2 == 2 and//g

%s/BEGIN_REMOVE_FOR_MATRIX\_.\{-}END_REMOVE_FOR_MATRIX/DELETED_CODE/g

normal gg
while search("UNCOMMENT_FOR_MATRIX", '', line("$")) > 0 " limit the search to end of file 
       " the last j is crucial or the cursor get up to the block and we have
       " an infinte loop
       normal V},cij

endwhile       

"Clang format
normal ==

