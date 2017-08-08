#!/usr/bin/env python

import os
import sys

import numpy as np
import HoomdAnalysis as ha


if len(sys.argv) != 4:
	print("\033[1;31mUsage is %s trajectory n_eq l_max\033[0m" % sys.argv[0])
	sys.exit()


file_traj  = os.path.realpath(sys.argv[1])

n_eq       = int(sys.argv[2])
l_max      = int(sys.argv[3])

path_traj  = os.path.dirname(file_traj)
a          = ha.Analyser(file_traj)

sph_aves   = a.accumulate(a.single_sph_aves, n_eq, log=True, l_max=l_max)
sph_aves   = np.mean(sph_aves, axis=0)

ops        = a.accumulate(a.nematic_q, n_eq)
op_ave     = np.mean(ops)

n_theta    = 100

theta_grid = np.linspace(0, np.pi, num=n_theta)
psi        = np.zeros([n_theta,2])

psi[:,0]   = theta_grid

for idx_theta,theta in enumerate(theta_grid):
	for l in range(l_max+1):
		if l % 2 == 0: psi[idx_theta,1] += np.real(sph_aves[l]*a.get_sph_harm(l, 0, theta, 0))

file_psi = "%s/psi_%d_%d.res" % (path_traj, n_eq, l_max)


np.savetxt(file_psi, psi)

print("Order parameter: %f, <Y20>: %f" % (op_ave, np.real(sph_aves[2]*np.sqrt(4*np.pi/5.))))
print("\033[1;32mpsi printed to '%s'\033[0m" % file_psi)

r_min  = 0.
r_max  = 12.

n_bins = 100

bins   = np.linspace(r_min, r_max, num=n_bins+1)

aves   = a.average(a.pair_sph_aves, n_eq, log=True, bins=bins, l_max=l_max)
aves   = aves.reshape([aves.shape[0], aves.shape[1]*aves.shape[2]*aves.shape[3]])

np.savetxt('real.txt', np.real(aves[:,:50]))
np.savetxt('imag.txt', np.imag(aves[:,:50]))
