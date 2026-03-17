#!/autofs/nccsopen-svm1_proj/mat269/baseline_matensembe_env/bin/python
import lammps
from mpi4py import MPI
import sys
import os
import numpy as np
from pathlib import Path

## LOAD INPUT PARAMETERS
lmp_input = sys.argv[1]
structure = sys.argv[2]
ff_filename = sys.argv[3]
control_filename = sys.argv[4]

lmp = lammps.lammps()
lmp.command(f'variable structure string {structure}')
lmp.command(f'variable ff_filename string {ff_filename}')
lmp.command(f'variable control_filename string {control_filename}')
lmp.file(lmp_input)
lmp.close()
