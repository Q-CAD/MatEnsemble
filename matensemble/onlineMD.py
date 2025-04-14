from matensemble.dynopro.ensemble import EnsembleDynamicsRunner
import numpy as np
import pandas as pd
import os

__author__ = "Soumendu Bagchi"
__package__= 'matensemble'



def ht_ensemble_md(candidate_parameters, initial_parameters_file=None,MD_tasks=56):        
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
    gen_sim_dir_list = []

    for i in range(N_sim):

        input_params['heat']['T_heat']=candidate_parameters[i]['Temp_K']
        input_params['lattice_strain']=candidate_parameters[i]['Strain']
        input_params['shear_strain']=candidate_parameters[i]['Shear_Strain']
        sim_args_list.append(json.dumps(input_params))
        gen_sim_dir_list.append(f'{candidate_parameters[i]["Temp_K"]}_K_{candidate_parameters[i]["Strain"]}_lattice_{candidate_parameters[i]["Shear_Strain"]}_shear')

        
    edr = EnsembleDynamicsRunner(sim_list=sim_list, \
                                sim_args_list=sim_args_list, \
                                sim_dir_list=gen_sim_dir_list, \
                                tasks_per_job=input_params['tasks_per_job'], \
                                cores_per_task=1,\
                                gpus_per_task=input_params['gpus_per_task'], \
                                write_restart_freq=input_params['write_restart_freq'], \
                                buffer_time=0.1)
    # run the ensemble of MD simulations
    edr.run()
   

    if 'return_output' in input_params.keys():
        return gather_data(gen_sim_dir_list, input_params['return_output'])
    else:
        
        return 


from matensemble.dynopro.task_lib.analysis_registry import AnalysisRegistry

@AnalysisRegistry.register('my_custom_analysis')
def my_custom_analysis(data, params):
    # Your custom analysis code here
    result = some_calculation(data, **params)
    
    # Save results
    with open(f'custom_analysis_{data.timestep}', 'w') as file:
        file.write(f'time-step result\n')
        file.write(f'{data.timestep} {result}')


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





