from XATOMS.postprocessors.ovito_calculators import OvitoCalculators
from XATOMS.postprocessors import compute_twist, compute_diffraction
import math

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
                
                data = OvitoCalculators(lmp_snapshot=lmp_snapshot)


                if 'dxa_analysis' in input_params.keys():

                        if input_params["dxa_analysis"]:
                                data.calculate_dxa(export_formats=input_params["dxa_output_formats"], dxa_line_sep=input_params["dxa_line_sep"], lattice="lattice")
                
                if input_params["full_trajectory_dump"]:
                        data.dump_trajectories(export_formats=input_params["trajectory_output_format"])
                
                if 'compute_twist' in input_params.keys():
                         
                        twist_angle = compute_twist.get_interlayer_twist(data=data, cutoff=input_params['compute_twist']['cutoff'], \
                                                                                     reference_particle_type=input_params['compute_twist']['reference_particle_type'], grid_resolution=input_params['compute_twist']['grid_resolution'], num_iter=input_params['compute_twist']['num_iter'])
                        with open(f'twist_{data.timestep}', 'w') as file:
                                file.write(f'time-step twist-angle\n')
                                file.write(f'{data.timestep} {twist_angle}')

                if 'target_window' in input_params['compute_twist'].keys():
                        
                        assert input_params['compute_twist']['grid_resolution']>1, f"Grid resolution has to be greater than 1 for for a multigrid coverage analysis"

                        from XATOMS.utils.stat import get_probability
                        prob = get_probability(twist_angle, target_window=input_params['compute_twist']['target_window'])
                        with open(f'coverage_probability_{data.timestep}', 'w') as file:
                                file.write(f'time-step coverage_prob min_twist max_twist\n')
                                file.write(f'{data.timestep} {prob} {input_params["compute_twist"]["target_window"][0]} {input_params["compute_twist"]["target_window"][1]}')
        
                
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