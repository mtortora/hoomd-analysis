import os
import sys

import gsd.pygsd
import gsd.hoomd

import numpy as np

cimport cython
cimport numpy as np

from scipy.special.cython_special cimport sph_harm


##################################
## Trajectory analyser class    ##
##################################

class Analyser():

	def __init__(self, file_traj):
		if not os.path.isfile(file_traj):
			print("\033[1;31mCouldn't find file '%s'\033[0m" % file_traj)
			sys.exit()

		data            = gsd.pygsd.GSDFile(open(file_traj, 'rb'))
		self.traj       = gsd.hoomd.HOOMDTrajectory(data)
		
		self.n_conf     = len(self.traj)
		self.n_part     = self.traj[0].particles.N


	# Fetch spherical harmonic of indices l,m
	def get_sph_harm(self, int l, int m, double theta, double phi):
		if abs(m) > l: raise ValueError("Must have |m| <= l")

		return sph_harm(m, l, phi, theta)


	# Get 1d indices from harmonic index pairs (l,m)
	def sph_idx(self, l, m):
		if abs(m) > l: raise ValueError("Must have |m| <= l")

		return int(l*(l+1)/2 + m)


	# x,y,z box dimensions
	def box_dims(self, snap): return snap.configuration.box[:3]


	# Main particle axes from quaternions
	def part_axis(self, snap):
		quats = snap.particles.orientation

		u0    = 2 * (quats[...,1]*quats[...,3] + quats[...,0]*quats[...,2])
		u1    = 2 * (quats[...,2]*quats[...,3] - quats[...,0]*quats[...,1])
		u2    = 1 - 2.*quats[...,1]**2 - 2.*quats[...,2]**2

		return np.asarray([u0,u1,u2]).T


	# Project 3xn vector(s) vecs in the frame of 3x3 matrix rot
	def proj_vec(self, vecs, rot): return np.dot(rot.T, vecs.T).T
	
	
	# Vectorial polar/azimuthal angles
	def sph_angs(self, vecs):
		vecs  /= np.linalg.norm(vecs, axis=-1, keepdims=True)
	
		thetas = np.asarray(np.arccos(vecs[...,2]), dtype=np.float32)
		phis   = np.asarray(np.arctan(vecs[...,1]/vecs[...,0]), dtype=np.float32)
	
		# Handle azimuthal phi degeneracy
		phis[vecs[...,0] < 0] += np.pi
	
		return np.asarray([thetas,phis], dtype=np.float32)


	# Nematic director from Q spectral analysis
	def nematic_q(self, snap, mode="ord"):
		qs          = np.zeros([self.n_part,3,3], dtype=np.float32)

		# Ensemble-averaged Q tensor
		u0, u1, u2  = self.part_axis(snap).T
		
		qs[:,0,0]   = u0*u0
		qs[:,1,0]   = u1*u0
		qs[:,2,0]   = u2*u0
		qs[:,0,1]   = u0*u1
		qs[:,1,1]   = u1*u1
		qs[:,2,1]   = u2*u1
		qs[:,0,2]   = u0*u2
		qs[:,1,2]   = u1*u2
		qs[:,2,2]   = u2*u2
		
		q           = 3/2. * np.mean(qs, axis=0) - 1/2. * np.eye(3)
		evals,evecs = np.linalg.eigh(q)

		# Sort eigenvalues/vectors in increasing order
		ordered     = evals.argsort()
	
		evals       = evals[ordered]
		evecs       = evecs[:,ordered]
		
		if np.linalg.det(evecs) < 0: evecs[:,0] *= -1
	
		if mode == "ord":   return evals[2]
		if mode == "frame": return np.asarray(evecs, dtype=np.float32)


	# Vectorised pairwise distance computations
	def p_dists(self, snap):
		box       = snap.configuration.box[:3]
		positions = snap.particles.position
		
		# Avoid double-counting
		unique    = np.triu_indices(self.n_part, k=1)
		
		vecs      = positions[:,None,...] - positions[None,...]
		vecs      = vecs[unique[0],unique[1],...]
		
		# Minimum image convention for PBCs
		for idx_axis in range(3):
			crossed_inf = (vecs[...,idx_axis] < -box[idx_axis]/2)
			crossed_sup = (vecs[...,idx_axis] >  box[idx_axis]/2)
			
			vecs[crossed_inf,idx_axis] += box[idx_axis]
			vecs[crossed_sup,idx_axis] -= box[idx_axis]
		
		dists = np.sqrt(np.sum(vecs**2, axis=1))
		
		return dists


	# Accumulates class property 'prop' over the last n_eq configurations
	def accumulate(self, prop, n_eq, **kwargs):
		if n_eq > self.n_conf: print("Warning: chosen equilibration time longer than trajectory")
		
		values = []
	
		for idx,snap in enumerate(self.traj[-n_eq:]): values.append(prop(snap, **kwargs))

		return np.asarray(values)


	# Averages class property 'prop' over the last n_eq configurations
	def average(self, prop, n_eq, **kwargs):
		if n_eq > self.n_conf: print("Warning: chosen equilibration time longer than trajectory")

		for idx,snap in enumerate(self.traj[-n_eq:]):
			if idx == 0: sum  = prop(snap, **kwargs)
			else:        sum += prop(snap, **kwargs)
		
			print("\033[1;34mProcessed %d out of %d configurations\033[0m"
			      % (idx+1, n_eq))

		return sum / n_eq


	# PDF from binned pairwise distances
	def g_hist(self, n_eq, n_bins=300, r_min=0, r_max=12):
		dims      = self.accumulate(self.box_dims, n_eq)
		dists     = self.accumulate(self.p_dists,  n_eq)

		# Discard distances greater than r_max/smallest box dimension
		max_dist  = np.min(dims)/2.
		r_max     = min(max_dist, r_max)
		
		dists     = dists[dists<r_max]

		vols      = dims[:,0]*dims[:,1]*dims[:,2]
		vol_ave   = np.mean(vols)

		hist,bins = np.histogram(dists, bins=np.linspace(r_min,r_max,num=n_bins+1,dtype=np.float32))

		# Renormalise histogram
		dr        = np.diff(bins)
		vol_bins  = 4 * np.pi * bins[1:]**2 * dr

		hist      = np.asfarray(hist, dtype=np.float32)
		
		hist     /= self.n_part*(self.n_part-1)/2. * n_eq
		hist     *= vol_ave/vol_bins

		return bins,hist


	# Single-particle spherical harmonic averages
	def single_sh_aves(self, snap, l_max=8):
		sh_aves     = np.zeros(l_max+1, dtype=np.float32)

		# Project particle axes in nematic frame and fetch spherical angles
		axes        = self.part_axis(snap)
		frame       = self.nematic_q(snap, mode="frame")

		axes_prj    = self.proj_vec(axes, frame)
		thetas,phis = self.sph_angs(axes_prj)
		
		for theta,phi in zip(thetas,phis):
			for l in range(0, l_max+1, 2): sh_aves[l] += np.real(self.get_sph_harm(l, 0, theta, phi))

		return sh_aves / self.n_part


	# Pair spherical harmonic averages
	def pair_sh_aves(self, snap, bins=np.linspace(0, 12, num=300+1), inds=None):
        # Cythonise input data
		cdef int        n_part = self.n_part
		cdef int        l_max  = np.max(inds)

		cdef float      r_min  = np.min(bins)
		cdef float      r_max  = np.max(bins)

		cdef Py_ssize_t n_tot  = len(inds)
		cdef Py_ssize_t n_bins = len(bins)-1

		cdef Py_ssize_t n_sh   = _sph_idx(l_max, l_max)+1

		cdef np.ndarray[np.int32_t,  ndim=2]   inds_c   = inds

		cdef np.ndarray[np.float32_t,ndim=1]   bins_c   = bins
		cdef np.ndarray[np.float32_t,ndim=1]   box_c    = snap.configuration.box[:3]

		cdef np.ndarray[np.float32_t,ndim=2]   cm_pos   = snap.particles.position

		cdef np.ndarray[np.float32_t,ndim=2]   axes     = self.part_axis(snap)
		cdef np.ndarray[np.float32_t,ndim=2]   frame    = self.nematic_q(snap, mode="frame")
		
		cdef np.ndarray[np.float32_t,ndim=2]   axes_prj = self.proj_vec(axes, frame)
		cdef np.ndarray[np.float32_t,ndim=2]   angs     = self.sph_angs(axes_prj)
        
		cdef np.ndarray[np.float32_t,ndim=1]   thetas   = angs[0]
		cdef np.ndarray[np.float32_t,ndim=1]   phis     = angs[1]

		# Set particle counter per spherical shell
		cdef np.ndarray[np.float32_t,ndim=1]   p_ctr    = np.zeros(n_bins, dtype=np.float32)

		# Spherical harmonics containers
		cdef np.ndarray[np.complex64_t,ndim=1] sh1      = np.zeros(n_sh, dtype=np.complex64)
		cdef np.ndarray[np.complex64_t,ndim=1] sh2      = np.zeros(n_sh, dtype=np.complex64)
		cdef np.ndarray[np.complex64_t,ndim=1] sh       = np.zeros(n_sh, dtype=np.complex64)
                
		cdef np.ndarray[np.float32_t,  ndim=2] sh_a     = np.zeros([n_bins,n_tot], dtype=np.float32)
                
		cdef np.ndarray[np.float32_t,  ndim=1] vec,vec_prj
		
		cdef int        l1,m1,l2,m2,l,m
		cdef Py_ssize_t idx_part1,idx_part2,idx_axis,idx_r,idx_h,idx,idx1,idx2
		
		cdef float theta1,phi1,theta2,phi2,theta,phi,r

		# Iterate over all particle pairs
		for idx_part1 in range(n_part):
			theta1,phi1 = thetas[idx_part1],phis[idx_part1]
			_set_sph_harms(sh1, l_max, theta1, phi1)

			for idx_part2 in range(idx_part1+1, n_part):
				vec = cm_pos[idx_part2,:] - cm_pos[idx_part1,:]
				
				for idx_axis in range(3):
					if vec[idx_axis] < -box_c[idx_axis]/2: vec[idx_axis] += box_c[idx_axis]
					if vec[idx_axis] >  box_c[idx_axis]/2: vec[idx_axis] -= box_c[idx_axis]
		
				r = np.linalg.norm(vec)

				if r_min < r < r_max:
					idx_r         = np.digitize(r, bins_c)-1
					p_ctr[idx_r] += 1
					
					vec_prj       = self.proj_vec(vec, frame)
					
					theta,phi     = self.sph_angs(vec_prj)
					theta2,phi2   = thetas[idx_part2],phis[idx_part2]

					_set_sph_harms(sh,  l_max, theta, phi)
					_set_sph_harms(sh2, l_max, theta2, phi2)

					# Average relevant sets of spherical harmonics
					for idx_h in range(n_tot):
						l1   = inds_c[idx_h,0]
						m1   = inds_c[idx_h,1]
						l2   = inds_c[idx_h,2]
						m2   = inds_c[idx_h,3]
						l    = inds_c[idx_h,4]
						m    = inds_c[idx_h,5]

						idx1 = _sph_idx(l1, m1)
						idx2 = _sph_idx(l2, m2)
						idx  = _sph_idx(l, m)

						sh_a[idx_r,idx_h] += (sh1[idx1]*sh2[idx2]*sh[idx]).real

		p_ctr[p_ctr==0] = 1.
		
		return sh_a / p_ctr[:,None]


##################################
## C-only convenience functions ##
##################################

# Get 1d indices from harmonic index pairs (l,m)
cdef Py_ssize_t _sph_idx(int l, int m) nogil: return int(l*(l+1)/2 + m)


# Fetch all harmonics up to rank l_max
cdef void _set_sph_harms(np.ndarray[np.complex64_t,ndim=1] sh, int l_max, double theta, double phi):
	cdef int           l,m,
	cdef Py_ssize_t    lmp,lmn

	cdef float complex slm

	for l in range(l_max+1):
		if l % 2 == 0:
			for m in range(l+1):
				lmp     = _sph_idx(l,  m)
				lmn     = _sph_idx(l, -m)

				slm     = sph_harm(m, l, phi, theta)

				sh[lmp] = slm
				sh[lmn] = (-1)**m * slm.conjugate()
