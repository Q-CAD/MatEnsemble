#!/usr/bin/python
"""
Parallel Hello World
"""

from mpi4py import MPI
import sys
# import omnitrace

# argument = sys.argv[1]

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

# sys.stdout.write(
#     "Hello, World! I am process %d of %d on %s.\n"
#     % (rank, size, name))

with open(f"{rank}.txt", "w") as file:
    file.write("Hello, World! I am process %d of %d on %s.\n" % (rank, size, name))
    # file.write(f"I will do something with the argument {argument} !\n")
