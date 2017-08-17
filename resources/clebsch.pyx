import  numpy as np
cimport numpy as np

from math import factorial


def cg(int j1, int j2, int j3, int m1, int m2, int m3):
	"""Calculates the Clebsch-Gordan coefficient <j1,m1,j2,m2|j3,m3>
		
	Parameters
	----------
	j1 : int
	Total angular momentum 1.
		
	j2 : int
	Total angular momentum 2.
		
	j3 : int
	Total angular momentum 3.
		
	m1 : int
	z-component of angular momentum 1.
		
	m2 : int
	z-component of angular momentum 2.
		
	m3 : int
	z-component of angular momentum 3.
		
	Returns
	-------
	cg_coeff : float
	Requested Clebsch-Gordan coefficient.
	"""
	
	cdef int          v,vmin,vmax
	cdef np.float32_t C,S


	if m3 != m1 + m2: return 0

	if ( (j3+j1-j2 < 0) | (j3-j1+j2 < 0) | (j1+j2-j3 < 0) ): return 0

	vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
	vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))
							
	C = np.sqrt((2.0 * j3 + 1.0) * factorial(j3 + j1 - j2) *
				factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) *
				factorial(j3 + m3) * factorial(j3 - m3) /
				(factorial(j1 + j2 + j3 + 1) *
				factorial(j1 - m1) * factorial(j1 + m1) *
				factorial(j2 - m2) * factorial(j2 + m2)))

	S = 0

	for v in range(vmin, vmax+1):
		S += (-1.0) ** (v + j2 + m2) / factorial(v) * \
			factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
			factorial(j3 - j1 + j2 - v) / factorial(j3 + m3 - v) / \
			factorial(v + j1 - j2 - m3)

	return C*S
