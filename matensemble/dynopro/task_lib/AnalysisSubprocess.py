from matensemble.dynopro.postprocessors.ovito_calculators import OvitoCalculators
from matensemble.dynopro.postprocessors import compute_twist, compute_diffraction
from matensemble.dynopro.task_lib.analysis_registry import AnalysisRegistry
import matensemble.dynopro.postprocessors.correlations as correlations
import math
import numpy as np

def AnalysisSubprocess(comm, input_params):

        nprocs = comm.Get_size()
        rank = comm.Get_rank()
        MAX_STRIDES = math.ceil(input_params["total_number_of_timesteps"]/input_params["i_o_freq"]/(nprocs - input_params['md_procs']))
        stride = 0 #

        while stride<MAX_STRIDES:
                
       # receive LAMMPS Snapshot


                timestep = (stride*(nprocs-input_params["md_procs"]) + (rank-input_params["md_procs"]))*input_params["i_o_freq"]

                # print (f'currently at rank {rank} and stride {stride} at time step: {timestep} BEFORE receving')
                
                if timestep<=input_params["total_number_of_timesteps"]:
                        lmp_snapshot = comm.recv(source=0)
                        # print (f'currently at rank {rank} and stride {stride} AFTER receving')
                else:
                        return
                
                # Run analysis for each snapshot
                
                data = OvitoCalculators(lmp_snapshot=lmp_snapshot, species=input_params['species'], serialize=True)

                if input_params.get('include_metadata', False):
                        metadata = {
                                "timestep": data.timestep,
                                "species": input_params['species'],
                                "lammps_input": input_params['lammps_input'],
                                "lammps_datafile": input_params['lammps_datafile'],
                                "force-field": input_params['force-field'],
                                "resurce_allocation": {"nnodes":input_params['nnodes'],
                                "gpus_per_node": input_params['gpus_per_node'],
                                "total_procs": input_params['total_procs'],
                                "md_procs": input_params['md_procs']},
                        }
                        if 'heat' in input_params.keys():
                                metadata['heat_temperature'] = input_params['heat']['T_heat']
                        if 'quench' in input_params.keys():
                                metadata['quench_temperature'] = input_params['quench']['T_quench']
                        
                        metadata['current_temperature'] = lmp_snapshot['temperature']
                        metadata['current_pressure'] = lmp_snapshot['pressure']
                        metadata['current_potential_energy'] = lmp_snapshot['potential_energy']
                else:
                        metadata = None

                if 'stream' in input_params.keys():
                        if 'host' not in input_params['stream'].keys():
                                raise ValueError("valid redis-host must be provided for data streaming")
                        if 'port' not in input_params['stream'].keys():
                                raise ValueError("valid redis-port must be provided for data streaming")
                        if 'namespace' not in input_params['stream'].keys():
                                raise ValueError("valid namespace must be provided for data streaming")
                        
                        from matensemble.redis.service import RedisService
                        rds = RedisService(host=input_params['stream']['host'], port=input_params['stream']['port'])
                else:
                        rds = None

                if 'dxa_analysis' in input_params.keys():

                        if input_params["dxa_analysis"]:
                                data.calculate_dxa(export_formats=input_params["dxa_output_formats"], dxa_line_sep=input_params["dxa_line_sep"], lattice="lattice")
                
                if input_params["full_trajectory_dump"]:
                        data.dump_trajectories(export_formats=input_params["trajectory_output_format"])

                        if metadata is not None:
                                metadata['dump'] = f'Ovito_dump.{data.timestep}.lmp'
                
                if 'compute_twist' in input_params.keys():
                         
                        twist_angle = compute_twist.get_interlayer_twist(data, cutoff=input_params['compute_twist']['cutoff'],
                                                                         reference_particle_type=input_params['compute_twist']['reference_particle_type'],
                                                                         grid_resolution=input_params['compute_twist']['grid_resolution'],
                                                                         num_iter=input_params['compute_twist']['num_iter'])

                        if rds is not None:
                                rds.register_on_stream(namespace=input_params['stream']['namespace'], key="twist_data", timestep=data.timestep, twist_data=twist_angle)
                        
                        if metadata is not None:
                                metadata['twist_angle'] = f'twist_{data.timestep}'

                        with open(f'twist_{data.timestep}', 'w') as file:
                                file.write(f'time-step twist-angle\n')
                                file.write(f'{data.timestep} {twist_angle}')

                        if 'target_window' in input_params['compute_twist'].keys():
                                
                                assert input_params['compute_twist']['grid_resolution']>1, f"Grid resolution has to be greater than 1 for for a multigrid coverage analysis"

                                from matensemble.dynopro.utils.stat import get_probability
                                prob = get_probability(twist_angle, target_window=input_params['compute_twist']['target_window'])
                                with open(f'coverage_probability_{data.timestep}', 'w') as file:
                                        file.write(f'time-step coverage_prob min_twist max_twist\n')
                                        file.write(f'{data.timestep} {prob} {input_params["compute_twist"]["target_window"][0]} {input_params["compute_twist"]["target_window"][1]}')

                if 'compute_xrd' in input_params.keys():
                        
                        if input_params['compute_xrd']:
                                filetag = f'XRD_{data.timestep}'
                                xrd_pattern = compute_diffraction.get_xrd_ovito(data, filetag)
                                if rds is not None:
                                        # Ensure XRD pattern is JSON-serializable before streaming
                                        if isinstance(xrd_pattern, np.ndarray):
                                                xrd_pattern_stream = xrd_pattern.tolist()
                                        else:
                                                xrd_pattern_stream = xrd_pattern
                                        rds.register_on_stream(
                                                namespace=input_params['stream']['namespace'],
                                                key="xrd_data",
                                                timestep=data.timestep,
                                                xrd_pattern=xrd_pattern_stream,
                                        )
                                if metadata is not None:
                                        metadata['xrd_pattern'] = f"{filetag}_xrd.dat"

                if 'compute_Laue_Diffraction' in input_params.keys():
                        
                        if input_params['compute_Laue_Diffraction']:
                                filetag = f'Laue_{data.timestep}'
                                compute_diffraction.get_laue_pattern(data, filetag)

                if 'compute_rdf' in input_params.keys():
                
                        
                        try:
                                rdf = correlations.compute_rdf(data, cutoff=input_params['compute_rdf']['cutoff'], number_of_bins=input_params['compute_rdf']['number_of_bins'], z_min=input_params['compute_rdf']['z_min'])

                                if rds is not None:
                                        rds.register_on_stream(namespace=input_params['stream']['namespace'], key="rdf_data", timestep=data.timestep, rdf=rdf.tolist())
                                
                                if metadata is not None:
                                        metadata['rdf'] = f'rdf_{data.timestep}.txt'
                                        
                                np.savetxt(f'rdf_{data.timestep}.txt', rdf, delimiter=' ')
                        except Exception as e:
                                print(f"Error computing RDF at timestep {data.timestep}: {e}")

                if 'compute_adf' in input_params.keys():


                        try:
                                adf = correlations.compute_adf(data, cutoff=input_params['compute_adf']['cutoff'], number_of_bins=input_params['compute_adf']['number_of_bins'], z_min=input_params['compute_adf']['z_min'])

                                if rds is not None:
                                        rds.register_on_stream(namespace=input_params['stream']['namespace'], key="adf_data", timestep=data.timestep, adf=adf.tolist())
                                
                                if metadata is not None:
                                        metadata['adf'] = f'adf_{data.timestep}.txt'
                                        
                                np.savetxt(f'adf_{data.timestep}.txt', adf, delimiter=' ')
                        
                        except Exception as e:
                                print(f"Error computing ADF at timestep {data.timestep}: {e}")
                
                if metadata is not None:
                        with open(f'metadata_{data.timestep}.json', 'w') as file:
                                import json
                                json.dump(metadata, file, indent=4)


                 # Execute registered analyses
                registered_analyses = AnalysisRegistry.get_registered_analyses()
                for analysis_name, analysis_func in registered_analyses.items():
                        if analysis_name in input_params:
                                if input_params[analysis_name].get('enabled'):
                                        analysis_data = analysis_func(data=data, params=input_params[analysis_name])
                                
                                        
      
                
                # if 'compute_Laue_Diffraction' in input_params.keys():
                        
                #         if input_params['compute_Laue_Diffraction']:
                #                 filetag = str(data.timestep)
                #                 compute_diffraction.get_laue_pattern(data, filetag)

                # if 'compute_xrd' in input_params.keys():

                #         if input_params['compute_xrd']:
                #                 filetag = str(lmp_snapshot.timestep)
                #                 compute_diffraction.get_xrd_pattern(data, filetag)

                        
                stride+=1


        return