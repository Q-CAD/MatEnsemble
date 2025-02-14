from matensemble.matflux import SuperFluxManager
import numpy as np
import pandas as pd
import os

__author__ = "Soumendu Bagchi"
__package__= 'matensemble'



def onlineMD(candidate_parameters, initial_parameters_file=None,MD_tasks=56, task_command='driver.py'):        
     # create a list of task indicators; In the following I use integers as task-IDs. 

    N_task = len(candidate_parameters)
    task_list = list(np.arange(N_task))

    # spceify the basic command/executable-filepath used to execute the task (you can skip any mpirun/srun prefixes, and also any *args, **kwargs at this point)

    # task_command = '/autofs/nccs-svm1_proj/cph162/python_environments/matensemble_env/lib/python3.11/site-packages/matensemble/matensemble/driver.py' #'mpi_helloworld.py' #make sure to make it executable by `chmod u+x <file.py>`

    # Now instatiate a task_manager object which is a Superflux Manager sitting on top of evey smaller Fluxlets

    if initial_parameters_file==None:
        master = SuperFluxManager(task_list, task_command, None, tasks_per_job=MD_tasks, cores_per_task=1, gpus_per_task=0, write_restart_freq=5)


    # Input argument list specific to each task in the sorted as task_ids
    import json

    if initial_parameters_file != None:
        try:
            with open(initial_parameters_file,'r') as file:
                input_params=json.load(file)
        except:
            print (f"Have to sepcify initial_paramters file to setup MD")

    master = SuperFluxManager(task_list, task_command, None, tasks_per_job=input_params['tasks_per_job'], cores_per_task=input_params['cores_per_task'], gpus_per_task=input_params['gpus_per_task'], write_restart_freq=input_params['write_restart_freq'])
    task_args_list = []
    gen_task_dir_list = []

    for i in range(N_task):
        input_params['heat']['T_heat']=candidate_parameters[i]['Temp_K']
        input_params['lattice_strain']=candidate_parameters[i]['Strain']
        input_params['shear_strain']=candidate_parameters[i]['Shear_Strain']
        task_args_list.append(json.dumps(input_params))
        gen_task_dir_list.append(f'{candidate_parameters[i]["Temp_K"]}_K_{candidate_parameters[i]["Strain"]}_lattice_{candidate_parameters[i]["Shear_Strain"]}_shear')

    # For multiple args per task each if the elements could be a list i.e. task_args_list = [['x0f','x14'],['xa9','xf3'],[]...]
    # finally execute the whole pool of tasks
    master.poolexecutor(task_args_list, buffer_time=0.1, task_dir_list=gen_task_dir_list)


    if 'return_output' in input_params.keys():
        return gather_data(gen_task_dir_list, input_params['return_output'])
    else:
        
        return 

def gather_data(dirs, params):

    output_data = {'dir':[], 'output':[]}
    
    for dir in dirs:
        data = pd.read_csv(f'{os.path.abspath(dir)}/{params["filename_pattern"]}{params["Tstart"]}', sep='\s+')
        for timestep in range(int(params['Tstart']+ params['Tstep']),int(params['Tend']+ params['Tstep']), int(params['Tstep'])):
            temp_df = pd.read_csv(f'{os.path.abspath(dir)}/{params["filename_pattern"]}{timestep}',  sep='\s+')
            data = pd.concat([data, temp_df])
        
        output_data['dir'].append(dir)
        output_data['output'].append(np.mean(data['twist-angle']))

    return output_data





