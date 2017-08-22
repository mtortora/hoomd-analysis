#!/usr/bin/env python

import os
import sys

import numpy as np
import HoomdAnalysis as ha


if len(sys.argv) != 3:
	print("\033[1;31mUsage is %s particle_folder n_eq\033[0m" % sys.argv[0])
	sys.exit()


dir_a  = str(sys.argv[1])
n_eq   = int(sys.argv[2])

if not os.path.isdir(dir_a):
	print("\033[1;31mCouldn't find directory '%s'\033[0m" % dir_a)
	sys.exit()

dirs_p = next(os.walk(dir_a))[1]
dirs_p.sort()

dir_a = dir_a.rstrip("/")
ps    = [float(x[-5:]) for x in dirs_p]

n_tot = len(dirs_p)
alpha = float(dir_a[-4:])

v0    = np.pi * 0.5**2 * (alpha + 4/3.*0.5)

p_sim = []
s_sim = []

for idx_p,dir_p in enumerate(dirs_p):
	print("\033[1;36mParsing trajectory %d out of %d\033[0m" % (idx_p+1, n_tot))
	
	file_traj = "%s/%s/trajectory.gsd" % (dir_a, dir_p)
	a         = ha.Analyser(file_traj)
	
	dims      = a.accumulate(a.box_dims,  n_eq)
	ops       = a.accumulate(a.nematic_q, n_eq)
	
	vols      = dims[:,0]*dims[:,1]*dims[:,2]
	etas      = a.n_part * v0/vols
	
	eta_ave   = etas.mean()
	op_ave    = ops.mean()
	
	p_adim    = ps[idx_p] * v0
	
	p_sim.append([eta_ave, p_adim, etas.min(), etas.max()])
	s_sim.append([eta_ave, op_ave, etas.min(), etas.max(), ops.min(), ops.max()])

file_p = "%s/p_%d.res" % (dir_a, n_eq)
file_s = "%s/s_%d.res" % (dir_a, n_eq)


np.savetxt(file_p, np.asarray(p_sim))
np.savetxt(file_s, np.asarray(s_sim))

print("\033[1;32mPressures printed to '%s'\033[0m" % file_p)
print("\033[1;32mOrder parameters printed to '%s'\033[0m" % file_s)
