#!/autofs/nccs-svm1_proj/cph162/python_environments/matensemble_env/bin/python

from mpi4py import MPI
import json
from XATOMS.task_lib.MDSubprocess import *
from XATOMS.task_lib.AnalysisSubprocess import *
import sys
import os

__author__ = "Soumendu Bagchi"
__package__= 'matensemble'


if __name__=='__main__':
        """
        "srun/mpirun -n <num_procs> python driver.py input_paramters.json"
        """
        
#        os.environ['HIP_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
        # Load input paramters from a json file
        try: 
                input_paramters_file = sys.argv[1]
        
        except Exception as e:
                raise SyntaxError("Correct systax: srun/mpirun -n <num_procs> python driver.py input_paramters.json")        
        try:

                with open(input_paramters_file,'r') as file:
                        input_params=json.load(file)
        except:
                input_params=json.loads(input_paramters_file)

        # Get rank details
        comm = MPI.COMM_WORLD
        me = comm.Get_rank()
        nprocs = comm.Get_size()
        #print ('total number of ranks'+str(nprocs))

        # create two subcommunicators
        if me < input_params['md_procs']:  color = 0
        else: color = 1
        split = comm.Split(color,key=0)

        # run the tasks
        if color == 0:
                MDSubprocess(split, comm, input_params=input_params)

        else:
                AnalysisSubprocess(comm, input_params=input_params)
                print (f"shutting down rank: {comm.Get_rank()}")

        # # #---- shutdown -------# # #
        comm.barrier()
        if me == 0:
                print ('Existing Simulation Environment')
        MPI.Finalize()
        
