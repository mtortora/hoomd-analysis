import os
import sys
import codecs
import argparse

import numpy as np

from hoomd import *
from hoomd import hpmc


# ========================
# ANSI escape sequences
# ========================

ANSIColor = {	
	"bla":"1;30m",
	"red":"1;31m",
	"gre":"1;32m",
	"yel":"1;33m",
	"blu":"1;34m",
	"pur":"1;35m",
	"cya":"1;36m",
	"whi":"1;37m"
}


# ========================
# Parse input parameters
# ========================

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("alpha",       help='particle aspect ratio',
					type=float,     metavar='L/D')
parser.add_argument("betaP",       help='equilibrium pressure (in kT per unit volume)',
					type=float,     metavar='P')
parser.add_argument("--n",         help='initial lattice size',
					type=int,       default=2,  metavar='n')
parser.add_argument("--N",         help='number of particles per unit cell side (for "cub" ant "ortho" init_types only)',
					type=int,       default=15, metavar='N')
parser.add_argument("--init_type", help='choose initial lattice type',
					type=str,       default="cub", choices=set(("sc","bcc","fcc","cub","ortho")))
parser.add_argument("--continue",  help='attempt to restart simulation from previous run',
					dest="restart", action="store_true")


# ========================
# Simulation manager
# ========================

class HoomdSim():
	
	def __init__(self, alpha, betaP, n, N, init_type, restart):
		"""Setup simulation"""
		
		# Use CPU only
		context.initialize('--mode=cpu')
		
		# Create working directory if needed
		path = "a_%.1f/p_%.3f" % (alpha,betaP)
		
		if comm.get_rank() == 0: 
			if not os.path.exists(path): os.makedirs(path)
		
		# Output files
		self.files_out = {'trajectory': "%s/trajectory.gsd" % path,
                          'analysis':   "%s/analysis.log"   % path}
		
		file_restart   = "%s/restart.gsd" % path

		# Try to restart where previous simulation left off, otherwise start from simple cubic crystal
		if restart:
			
			if os.path.exists(file_restart):
				system = init.read_gsd(filename=file_restart)
				
				self.log("Starting from configuration %s - ignoring optional input arguments"
						 % file_restart, 'blu')
				self.log("%s" % system.particles, 'pur')

			else:
				self.log("Could not find file %s - starting fresh" % file_restart, 'blu')
				self.init_lat(alpha, n, N, init_type)

		else:
			self.init_lat(alpha, n, N, init_type)

			# Avoid overwriting existing output
			for key,file in self.files_out.items():
				
				if os.path.exists(file):
					idx_file  = 2
					
					name, ext = os.path.splitext(file)
					new_file  = name + "%s" + ext
					
					while os.path.exists(new_file % idx_file): idx_file += 1
				
					new_file            = new_file % idx_file
					self.files_out[key] = new_file
		
					self.log("Found file %s - writing %s to %s instead"
						     % (file,key,new_file), 'cya')

				else: self.log("Writing %s to %s" % (key,file), 'cya')
		
		# Entropy-harvested rng seed
		rng_seed = os.urandom(4)
		rng_seed = int(codecs.encode(rng_seed, 'hex'), 16)

		# Setup integrator and particle model
		vert     = [(0.,0.,-alpha/2.), (0.,0.,alpha/2.)]
		
		mc       = hpmc.integrate.convex_spheropolyhedron(seed=rng_seed, d=0.5, a=0.025)
		mc.shape_param.set('A', vertices=vert, sweep_radius=0.5, ignore_statistics=False)

		# Setup NPT simulation box
		boxmc = hpmc.update.boxmc(mc, betaP=betaP, seed=rng_seed+1)
		
		boxmc.length(delta=0.005, weight=1)
		boxmc.aspect(delta=0.01, weight=1)
		
		# Recovery configuration file
		self.gsd_restart = dump.gsd(filename=file_restart, group=group.all(), truncate=True,
									period=100, phase=0)


	def log(self, str, color):
		"""Thread-safe logger"""
		
		# ANSI colors
		CSI   = "\033["
		reset = CSI+"0m"
		
		if comm.get_rank() == 0:
			print(CSI + ANSIColor[color] + str + reset)
			sys.stdout.flush()
			

	def init_lat(self, alpha, n, N, init_type):
		"""Setup initial lattice"""
					
		if init_type == "sc":
			init.create_lattice(unitcell=lattice.sc(a=alpha+1.), n=n);
		
			self.log("Using simple cubic cell of dimension %.1f (total %d particles)"
					 % (alpha+1.,n**3), 'pur')
					
		elif init_type == "bcc":
			init.create_lattice(unitcell=lattice.bcc(a=alpha+1.), n=n);
			
			self.log("Using body-centered cubic cell of dimension %.1f (total %d particles)"
					 % (alpha+1.,2*n**3), 'pur')

		elif init_type == "fcc":
			init.create_lattice(unitcell=lattice.fcc(a=alpha+1.), n=n);
		
			self.log("Using face-centered cubic cell of dimension %.1f (total %d particles)"
					 % (alpha+1.,4*n**3), 'pur')

		elif init_type == "cub":
			N   = min(N, int(0.9*alpha))
			
			x_p = np.linspace(0., alpha+1., num=N, endpoint=False)
			y_p = np.linspace(0., alpha+1., num=N, endpoint=False)

			ps  = np.vstack(np.meshgrid(x_p,y_p,0.)).reshape(3,-1).T
			
			uc  = lattice.unitcell(N=N**2,
								   a1=[alpha+1.,0,0], a2=[0,alpha+1.,0], a3=[0,0,alpha+1.],
								   position=ps)
			init.create_lattice(unitcell=uc, n=n)
				
			self.log("Using custom cubic cell of dimension %.1f (total %d particles)"
					 % (alpha+1.,N**2*n**3), 'pur')

		elif init_type == "ortho":
			x_p = np.arange(N) * 2.
			y_p = np.arange(N) * 2.
			
			ps  = np.vstack(np.meshgrid(x_p,y_p,0.)).reshape(3,-1).T
			
			uc  = lattice.unitcell(N=N**2,
								   a1=[2*N,0,0], a2=[0,2*N,0], a3=[0,0,alpha+1.],
								   position=ps)
			init.create_lattice(unitcell=uc, n=n)
								   
			self.log("Using custom orthorombic cell of dimensions %.1f,%.1f,%.1f (total %d particles)"
					 % (2*N,2*N,alpha+1.,N**2*n**3), 'pur')
	
	
	def run(self, N):
		"""Run simulation"""
		
		# Write logs/trajectory
		analyze.log(filename=self.files_out['analysis'], quantities=['num_particles', 'volume'],
					period=100, phase=0)
		dump.gsd(self.files_out['trajectory'], group=group.all(),
				 period=1000, phase=0)

		# Restartable run over N steps
		run_upto(N);


	def __exit__(self, etype, value, traceback):
	
		# Save last frame and exit
		self.gsd_restart.write_restart()


# ********************** #

if __name__ == "__main__":
	sim = HoomdSim(**vars(parser.parse_args()))
	
	sim.run(5e6)
