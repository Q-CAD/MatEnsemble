from mpi4py import MPI
import json
from matensemble.dynopro.task_lib.MDSubprocess import *
from matensemble.dynopro.task_lib.AnalysisSubprocess import *
import sys
import os

__author__ = "Soumendu Bagchi"
__package__ = 'matensemble'

def online_dynamics(input_params_source):
    """
    Run MD simulation and analysis using MPI communication.
    
    Args:
        input_params_source (str or dict): Either path to JSON file or JSON string or dict containing parameters
    
    Returns:
        None
        
    Raises:
        SyntaxError: If input parameters are not properly provided
        ValueError: If input parameters are invalid
    """
    # Load input parameters
    if isinstance(input_params_source, dict):
        input_params = input_params_source
    elif isinstance(input_params_source, str):
        try:
            # Try reading as JSON file
            with open(input_params_source, 'r') as file:
                input_params = json.load(file)
        except FileNotFoundError:
            # Try parsing as JSON string
            try:
                input_params = json.loads(input_params_source)
            except json.JSONDecodeError:
                raise ValueError("Input must be either a valid JSON file path, JSON string, or dictionary")
    else:
        raise ValueError("Input must be either a valid JSON file path, JSON string, or dictionary")

    # Get rank details
    comm = MPI.COMM_WORLD
    me = comm.Get_rank()
    nprocs = comm.Get_size()

    # Create two subcommunicators
    color = 0 if me < input_params['md_procs'] else 1
    split = comm.Split(color, key=0)

    # Run the tasks
    if color == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(me % int(os.environ.get('SLURM_GPUS_PER_NODE')))
        os.environ['HIP_VISIBLE_DEVICES'] = str(me % int(os.environ.get('SLURM_GPUS_PER_NODE')))
        MDSubprocess(split, comm, input_params=input_params)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['HIP_VISIBLE_DEVICES'] = ''
        AnalysisSubprocess(comm, input_params=input_params)
        print(f"shutting down rank: {comm.Get_rank()}")

    # Shutdown
    comm.barrier()
    if me == 0:
        print('Exiting Simulation Environment')
    MPI.Finalize()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run online dynamics simulation')
    parser.add_argument('input_file', type=str, 
                       help='Path to JSON input file or JSON string')
    args = parser.parse_args()
    
    print ('Starting onine dynamics now . . . . ')
    online_dynamics(args.input_file)


