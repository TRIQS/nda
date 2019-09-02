" Clean after START  
normal /STARTjdGo

 " Find the pattern -> "t register
normal /PATTERN12jV}"ty	

" a list of couple
let L= [["first","0"], ["second","1"], ["third","2"], ["fourth","3"], ["fifth","4"], ["sixth","5"], ["seventh","6"], ["eighth","7"], ["ninth","8"]]

for [a,b] in L
 " paste the pattern at the end 
 normal G"tP 
 " replace XXX and NNN. { : put cursor up before second replace
 execute "normal! :.,$s/XXX/".a."/g"
 execute "normal! {:.,$s/NNN/".b."/g"
endfor  

"Clang format
normal ==



    
