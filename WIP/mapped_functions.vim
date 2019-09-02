" Clean after START  
normal /STARTjdGo

function! DoOnePattern(num)

 " Find the pattern -> "t register
 execute "normal! /PATTERN".a:num."2jV}\"ty"	

 " List into "d register
 execute "normal! /LIST".a:num."w\"dy$"

 " for all element in the list, paste the "t at the end of the file and replace from there till the end X by i
 for i in split(@d)
  execute "normal! G\"tP:.,$s/X/".i."/g"
 endfor  

endfunction


call DoOnePattern(1)
call DoOnePattern(2)
call DoOnePattern(3)

"Clang format
normal ==

