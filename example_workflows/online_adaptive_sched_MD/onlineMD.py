from matensemble.dynopro.ensemble import EnsembleDynamicsRunner
import numpy as np
import pandas as pd
import os



def onlineMD(candidate_parameters, initial_parameters_file=None):        
     # create a list of task indicators; In the following I use integers as task-IDs. 

    N_sim = len(candidate_parameters)
    sim_list = list(np.arange(N_sim))

    # Input argument list specific to each task in the sorted as task_ids
    import json

    if initial_parameters_file != None:
        try:
            with open(initial_parameters_file,'r') as file:
                input_params=json.load(file)
        except:
            print (f"Have to sepcify initial_paramters file to setup MD")


    sim_args_list = []
    sim_dir_list = []

    for i in range(N_sim):
        input_params['heat']['T_heat']=candidate_parameters[i]['Temp_K']
        input_params['lattice_strain']=candidate_parameters[i]['Strain']
        input_params['shear_strain']=candidate_parameters[i]['Shear_Strain']
        sim_args_list.append(json.dumps(input_params))
        sim_dir_list.append(f'{candidate_parameters[i]["Temp_K"]}_K_{candidate_parameters[i]["Strain"]}_lattice_{candidate_parameters[i]["Shear_Strain"]}_shear')

    edr = EnsembleDynamicsRunner(sim_list=sim_list, \
                sim_args_list=sim_args_list, \
                sim_dir_list=sim_dir_list, \
                tasks_per_job=input_params['total_procs'], \
                cores_per_task=1, \
                gpus_per_task=0, \
                write_restart_freq=10, \
                buffer_time=0.5, adaptive=True)

    edr.run()

    if 'return_output' in input_params.keys():
        return gather_data(sim_dir_list, input_params['return_output'])
    else:
        
        return 

def gather_data(dirs, params):

    output_data = {'dir':[], 'output':[]}
    
    for dir in dirs:
        data = pd.read_csv(f'{os.path.abspath(dir)}/{params["filename_pattern"]}{params["Tstart"]}', sep='\s+',
                           converters={'twist-angle': lambda x: float(x.strip('[]'))})  # Convert string lists to float
    
        for timestep in range(int(params['Tstart']+ params['Tstep']),int(params['Tend']+ params['Tstep']), int(params['Tstep'])):
            temp_df = pd.read_csv(f'{os.path.abspath(dir)}/{params["filename_pattern"]}{timestep}',  sep='\s+',
                                converters={'twist-angle': lambda x: float(x.strip('[]'))})  # Convert string lists to float
            data = pd.concat([data, temp_df])
        
        output_data['dir'].append(dir)
        output_data['output'].append(np.mean(data['twist-angle']))

    return output_data


if __name__=="__main__":

    n_param = 15
    temperature_data = np.linspace(1500,3000, n_param)
    lattice_strain_data = np.linspace(1.00,1.05, n_param)
    shear_strain_data = np.linspace(1.00, 1.05, n_param)
    grid = np.array(np.meshgrid(temperature_data, lattice_strain_data, shear_strain_data)).T.reshape(-1,3)    
    params_dict = []
    num_samples = len(grid)

    import sys
    partial_samples_init = int(sys.argv[1])
    partial_samples_final = int(sys.argv[2])

    for ic in range(partial_samples_init, partial_samples_final): 
        params_dict.append({'Temp_K': grid[ic,0], 'Strain': grid[ic,1], 'Shear_Strain': grid[ic,2]})

    onlineMD(candidate_parameters=params_dict,initial_parameters_file='input_paramters.json')


