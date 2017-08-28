cimport cython

import  numpy as np
cimport numpy as np

from math import factorial


# Base Clebsch-Gordan computation
cdef double _cg(int j1, int j2, int j3, int m1, int m2, int m3):
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
	cg_coeff : double
	Requested Clebsch-Gordan coefficient.
	"""

	cdef int    v,vmin,vmax
	cdef double c,s,cg_coeff

	if m3 != m1 + m2: return 0.
	if ( (j3+j1-j2 < 0) | (j3-j1+j2 < 0) | (j1+j2-j3 < 0) ): return 0.

	vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
	vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))
							
	c = np.sqrt((2. * j3 + 1.) * factorial(j3 + j1 - j2) *
				factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) *
				factorial(j3 + m3) * factorial(j3 - m3) /
				(factorial(j1 + j2 + j3 + 1) *
				 factorial(j1 - m1) * factorial(j1 + m1) *
				 factorial(j2 - m2) * factorial(j2 + m2)))

	s = 0.

	for v in range(vmin, vmax+1):
		s += (-1.) ** (v + j2 + m2) / factorial(v) * \
			factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
			factorial(j3 - j1 + j2 - v) / factorial(j3 + m3 - v) / \
			factorial(v + j1 - j2 - m3)

	cg_coeff = c*s

	return cg_coeff


# Get 1d indices from harmonic index pairs (l,m)
cdef Py_ssize_t _sph_idx(int l, int m) nogil: return int(l*(l+1)/2 + m)


# Renormalised Clebsch-Gordan coefficients
cdef double _cg_norm(int l, int lp, int lpp, int m, int mp, int mpp):
	cdef double coeff1 = _cg(lpp, lp, l, mpp, mp, m)
	cdef double coeff2 = _cg(lpp, lp, l, 0,   0,  0)
        
	return np.sqrt((2.*lpp+1)*(2.*lp+1)/(4*np.pi*(2.*l+1))) * coeff1*coeff2


# Tabulate Clebsch-Gordan gamma coefficients up to rank l_max
cpdef np.ndarray[np.float64_t,ndim=3] cg_tabulate(int l_max):
	cdef int        l,lp,lpp,m,mp,mpp
	
	cdef Py_ssize_t idx1,idx2,idx3
	cdef Py_ssize_t n_sh = _sph_idx(l_max, l_max)+1

	cdef np.ndarray[np.float64_t,ndim=3] cgs = np.zeros([n_sh,n_sh,n_sh], dtype=np.float64)

	for l in range(l_max+1):
		for lp in range(l_max+1):
			for lpp in range(l_max+1):
				if ( (l % 2 == 0) & (lp % 2 == 0) & (lpp % 2 == 0) ):
					for m in range(-l, l+1):
						for mp in range(-lp, lp+1):
							for mpp in range(-lpp, lpp+1):
								idx1 = _sph_idx(l,m)
								idx2 = _sph_idx(lp,mp)
								idx3 = _sph_idx(lpp,mpp)
                                    
								cgs[idx1,idx2,idx3] = _cg_norm(l, lp, lpp, m, mp, mpp)

	return cgs
