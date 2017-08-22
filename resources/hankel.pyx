cimport cython

import  numpy as np
cimport numpy as np

from scipy.special import spherical_jn


# Computes Hankel transform for h matrix with coefficients inds
cpdef np.ndarray[np.float32_t,ndim=2] hk_t(np.ndarray[np.float32_t,ndim=2] h,
                                           np.ndarray[np.int32_t,  ndim=2] inds,
                                           np.ndarray[np.float32_t,ndim=1] r_bins,
                                           np.ndarray[np.float32_t,ndim=1] ks):

	cdef int l_max      = inds.max()

	cdef Py_ssize_t n_h = h.shape[1]
	cdef Py_ssize_t n_k = ks.shape[0]

	cdef Py_ssize_t n_r = r_bins.shape[0]-1

	cdef np.ndarray[np.float32_t,ndim=1] rs   = r_bins[1:]
	cdef np.ndarray[np.float32_t,ndim=1] drs  = np.diff(r_bins)

	cdef np.ndarray[np.float32_t,ndim=2] hk   = np.zeros([n_k,n_h], dtype=np.float32)

	# Tabulate Bessel functions
	cdef np.ndarray[np.float32_t,ndim=3] c_bs = _bessel(l_max, rs, ks)

	cdef int l
	cdef Py_ssize_t idx_k,idx_h,idx_l,idx_r

	for idx_k in range(n_k):
		for idx_h in range(n_h):
			l     = inds[idx_h,4]
			idx_l = l//2

			# Bessel radial integral
			for idx_r in range(n_r):
				hk[idx_k,idx_h] += h[idx_r,idx_h]*c_bs[idx_r,idx_k,idx_l] * drs[idx_r]

	hk *= 4*np.pi

	return hk


# Computes Bessel spherical functions of the first kind on r and k grid up to rank l_max
cdef np.ndarray[np.float32_t,ndim=3] _bessel(int l_max, 
                                             np.ndarray[np.float32_t,ndim=1] rs,
                                             np.ndarray[np.float32_t,ndim=1] ks):

	cdef Py_ssize_t n_r = rs.shape[0]
	cdef Py_ssize_t n_k = ks.shape[0]

	cdef np.ndarray[np.float32_t,ndim=3] c_bs = np.zeros([n_r,n_k,l_max//2+1], dtype=np.float32)

	cdef int   l
	cdef float k

	cdef Py_ssize_t idx_k,idx_l

	for idx_k in range(n_k):
		k = ks[idx_k]

		for l in range(l_max+1):
			if l % 2 == 0:
				idx_l = l//2

				# Spherical Bessel function with radial normalisation
				c_bs[:,idx_k,idx_l] = (-1.)**idx_l * spherical_jn(l, k*rs) * rs**2

	return c_bs
