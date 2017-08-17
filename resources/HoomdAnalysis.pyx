#!/usr/bin/env python

import os
import sys

import gsd.pygsd
import gsd.hoomd

import numpy as np

cimport numpy as np
cimport scipy.special.cython_special as cs


# Trajectory analyser class
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
	def get_sph_harm(self, long l, long m, double theta, double phi):
		if abs(m) > l: raise ValueError("Must have |m| <= l")
			
		return cs.sph_harm(m, l, phi, theta)
	
	
	# Get 1d indices from harmonic index pairs (l,m)
	def sph_idx(self, long l, long m):
		if abs(m) > l: raise ValueError("Must have |m| <= l")
	
		return long(l*(l+1)/2 + m)


	# Fetch all harmonics up to rank l_max
	def get_sph_harms(self, long l_max, double theta, double phi):
		cdef int n_sh = self.sph_idx(l_max, l_max)+1
		cdef np.ndarray[np.complex64_t, ndim=1] sh = np.zeros(n_sh, dtype=np.complex64)
		
		cdef long           l,m,lmp,lmn
		cdef np.complex64_t slm
        
		for l in range(l_max+1):
			
			# Discard odd ls
			if l % 2 == 0:
				for m in range(l+1):
					lmp     = self.sph_idx(l,  m)
					lmn     = self.sph_idx(l, -m)

					slm     = cs.sph_harm(m, l, phi, theta)

					sh[lmp] = slm
					sh[lmn] = (-1)**m * np.conj(slm)

		return sh


	# x,y,z box dimensions
	def box_dims(self, snap): return snap.configuration.box[:3]


	# Main particle axes from quaternions
	def part_axis(self, quats):
		u0 = 2 * (quats[...,1]*quats[...,3] + quats[...,0]*quats[...,2])
		u1 = 2 * (quats[...,2]*quats[...,3] - quats[...,0]*quats[...,1])
		u2 = 1 - 2.*quats[...,1]**2 - 2.*quats[...,2]**2

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
		quats       = snap.particles.orientation

		# Ensemble-averaged Q tensor
		u0, u1, u2  = self.part_axis(quats).T
		
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
	
		if mode == "ord":   return evals[-1]
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
	def accumulate(self, prop, n_eq, log=False, **kwargs):
		if n_eq > self.n_conf: print("Warning: chosen equilibration time longer than trajectory")
		
		values = []
	
		for idx,snap in enumerate(self.traj[-n_eq:]):
			values.append(prop(snap, **kwargs))

			if log:
				if (idx+1) % 100 == 0: print("\033[1;34mProcessed %d out of %d configurations\033[0m"
											 % (idx+1, n_eq))

		return np.asarray(values)


	# Averages class property 'prop' over the last n_eq configurations
	def average(self, prop, n_eq, log=False, **kwargs):
		if n_eq > self.n_conf: print("Warning: chosen equilibration time longer than trajectory")

		for idx,snap in enumerate(self.traj[-n_eq:]):
			if idx == 0: sum  = prop(snap, **kwargs)
			else:        sum += prop(snap, **kwargs)
		
			if log: print("\033[1;34mProcessed %d out of %d configurations\033[0m"
						  % (idx+1, n_eq))

		return sum / n_eq


	# PDF from binned pairwise distances
	def g_hist(self, n_eq, n_bins=300, r_min=0, r_max=12):
		dims      = self.accumulate(self.box_dims, n_eq)
		dists     = self.accumulate(self.p_dists,  n_eq, log=True)

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
		quats       = snap.particles.orientation
		sh_aves     = np.zeros(l_max+1, dtype=np.complex64)

		# Project particle axes in nematic frame and fetch spherical angles
		axes        = self.part_axis(quats)
		frame       = self.nematic_q(snap, mode="frame")

		axes_prj    = self.proj_vec(axes, frame)
		thetas,phis = self.sph_angs(axes_prj)
		
		for theta,phi in zip(thetas,phis):
			for l in range(l_max+1):
				if l % 2 == 0: sh_aves[l] += self.get_sph_harm(l, 0, theta, phi)

		return sh_aves / self.n_part


	# Pair spherical harmonic averages
	def pair_sh_aves(self, snap, bins=np.linspace(0, 12, num=300+1), l_max=8):
		cdef int          n_part                         = self.n_part
		cdef int          n_bins                         = len(bins)-1
        
		cdef int          n_sh                           = self.sph_idx(l_max, l_max)+1

		cdef np.float32_t r_min                          = np.min(bins)
		cdef np.float32_t r_max                          = np.max(bins)

		cdef np.ndarray[np.float32_t, ndim=1]   box      = snap.configuration.box[:3]

		cdef np.ndarray[np.float32_t, ndim=2]   cm_pos   = snap.particles.position
		cdef np.ndarray[np.float32_t, ndim=2]   quats    = snap.particles.orientation
		
		# r-dependent coefficient (l1,m1,l2,m2,m,l) can be accessed as sh_aves[:,self.sph_idx(l1,m1),self.sph_idx(l2,m2),self.sph_idx(l,m)]
		cdef np.ndarray[np.complex64_t, ndim=1] sh1      = np.zeros(n_sh, dtype=np.complex64)
		cdef np.ndarray[np.complex64_t, ndim=1] sh2      = np.zeros(n_sh, dtype=np.complex64)
		cdef np.ndarray[np.complex64_t, ndim=1] sh       = np.zeros(n_sh, dtype=np.complex64)
        
		cdef np.ndarray[np.complex64_t, ndim=4] sh_aves  = np.zeros([n_bins,n_sh,n_sh,n_sh], dtype=np.complex64)

		# Set particle counter per spherical shell
		cdef np.ndarray[np.float32_t, ndim=1]   p_ctr    = np.zeros(n_bins, dtype=np.float32)
		
		# Project particle axes in nematic frame and fetch spherical angles
		cdef np.ndarray[np.float32_t, ndim=2]   axes     = self.part_axis(quats)
		cdef np.ndarray[np.float32_t, ndim=2]   frame    = self.nematic_q(snap, mode="frame")
		
		cdef np.ndarray[np.float32_t, ndim=2]   axes_prj = self.proj_vec(axes, frame)
		cdef np.ndarray[np.float32_t, ndim=2]   angs     = self.sph_angs(axes_prj)
        
		cdef np.ndarray[np.float32_t, ndim=1]   thetas   = angs[0]
		cdef np.ndarray[np.float32_t, ndim=1]   phis     = angs[1]
		
		cdef np.ndarray[np.float32_t, ndim=1]   vec, vec_prj
		
		cdef int          idx_part1,idx_part2,idx_axis,idx_r
		cdef np.float32_t theta1,phi1,theta2,phi2,theta,phi,r
		
		for idx_part1 in range(self.n_part):
			theta1,phi1 = thetas[idx_part1],phis[idx_part1]
			sh1         = self.get_sph_harms(l_max, theta1, phi1)

			for idx_part2 in range(idx_part1+1, self.n_part):
				vec = cm_pos[idx_part2,:] - cm_pos[idx_part1,:]

				# Minimum image convention for PBCs
				for idx_axis in range(3):
					if vec[idx_axis] < -box[idx_axis]/2: vec[idx_axis] += box[idx_axis]
					if vec[idx_axis] >  box[idx_axis]/2: vec[idx_axis] -= box[idx_axis]
		
				r = np.linalg.norm(vec)

				if r_min < r < r_max:
					idx_r         = np.digitize(r, bins)-1
					p_ctr[idx_r] += 1
					
					vec_prj       = self.proj_vec(vec, frame)
					
					theta,phi     = self.sph_angs(vec_prj)
					theta2,phi2   = thetas[idx_part2],phis[idx_part2]
		
					sh            = self.get_sph_harms(l_max, theta, phi)
					sh2           = self.get_sph_harms(l_max, theta2, phi2)

					sh_aves[idx_r,...] += np.einsum("i,j,k", sh1, sh2, sh)

		p_ctr[p_ctr==0] = 1.
		
		return sh_aves.conj() / p_ctr[:,None,None,None]
