import numpy as np
    

def myticks(x,pos):
     
  if x == 0: return "$0$"

  exponent = int(np.log10(np.abs(x)))
  coeff = x/10**(exponent-2)

  return r"${:2.0f} \times 10^{{ {:2d} }}$".format(coeff,exponent-2)
    
