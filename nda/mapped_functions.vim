" Clean 
normal /STARTjdGo

" Find the pattern -> "t register
normal /PATTERN12jV}"ty	

" List into "d register
normal /LIST1w"dy$

" for all element in the list, paste the "t at the end and replace
for i in split(@d)
  execute "normal! G\"tP:.,$s/X/".i."/ge"
endfor  

" SAME for 2 and 3

normal /PATTERN22jV}"ty	
normal /LIST2w"dy$

for i in split(@d)
  execute "normal! G\"tP:.,$s/X/".i."/ge"
endfor  


normal /PATTERN32jV}"ty	
normal /LIST3w"dy$

for i in split(@d)
  execute "normal! G\"tP:.,$s/X/".i."/ge"
endfor  

normal ==

