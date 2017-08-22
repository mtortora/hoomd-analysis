import os
import sys
import subprocess

from distutils.core import setup
from Cython.Build import cythonize

import numpy as np


args = sys.argv[1:]

# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
	sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")

# Only build for 64-bit target
os.environ['ARCHFLAGS'] = "-arch x86_64"

# Set up extension and build
setup(ext_modules=cythonize("*.pyx", compiler_directives={'boundscheck': False, 'wraparound': False}), include_dirs=[np.get_include()])

# Cleanup
if "clean" in args:
	print("deleting Cython-generated source files")

	subprocess.Popen("rm -rf *.c", shell=True, executable="/bin/bash")
